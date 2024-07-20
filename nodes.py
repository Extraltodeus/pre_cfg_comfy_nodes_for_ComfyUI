import torch
from math import ceil
from copy import deepcopy
import comfy.model_patcher
from comfy.sampler_helpers import convert_cond
from comfy.samplers import calc_cond_batch, encode_model_conds
from comfy.ldm.modules.attention import optimized_attention_for_device
from nodes import ConditioningConcat, ConditioningSetTimestepRange
import comfy.model_management as model_management
from comfy.latent_formats import SDXL as SDXL_Latent

SDXL_Latent = SDXL_Latent()
sdxl_latent_rgb_factors = SDXL_Latent.latent_rgb_factors
ConditioningConcat = ConditioningConcat()
ConditioningSetTimestepRange = ConditioningSetTimestepRange()
default_attention = optimized_attention_for_device(model_management.get_torch_device())

weighted_average = lambda tensor1, tensor2, weight1: (weight1 * tensor1 + (1 - weight1) * tensor2)
selfnorm = lambda x: x / x.norm()

class pre_cfg_perp_neg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "neg_scale": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,  "step": 1/10, "round": 0.01}),
                                "set_context_length" : ("BOOLEAN", {"default": False}),
                                "context_length": ("INT", {"default": 1,  "min": 1, "max": 100,  "step": 1}),
                                "start_at_sigma": ("FLOAT", {"default": 15,  "min": 0.0, "max": 1000.0, "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, clip, neg_scale, set_context_length, context_length, start_at_sigma, end_at_sigma):
        empty_cond, pooled = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        nocond = [[empty_cond, {"pooled_output": pooled}]]
        if context_length > 1 and set_context_length:
            short_nocond = deepcopy(nocond)
            for x in range(context_length - 1):
                (nocond,) = ConditioningConcat.concat(nocond, short_nocond)
        nocond = convert_cond(nocond)

        @torch.no_grad()
        def pre_cfg_perp_neg_function(args):
            conds_out = args["conds_out"]
            noise_pred_pos = conds_out[0]

            if args["sigma"][0] > start_at_sigma or args["sigma"][0] <= end_at_sigma or not torch.any(conds_out[1]):
                return conds_out

            noise_pred_neg = conds_out[1]

            model_options = args["model_options"]
            timestep = args["timestep"]
            model = args["model"]
            x = args["input"]

            nocond_processed = encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")
            (noise_pred_nocond,) = calc_cond_batch(model, [nocond_processed], x, timestep, model_options)

            pos = noise_pred_pos - noise_pred_nocond
            neg = noise_pred_neg - noise_pred_nocond

            perp = neg - ((torch.mul(neg, pos).sum())/(torch.norm(pos)**2)) * pos
            perp_neg = perp * neg_scale

            conds_out[0] = noise_pred_nocond + pos
            conds_out[1] = noise_pred_nocond + perp_neg

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_perp_neg_function)
        return (m, )

@torch.no_grad()
def normalize_adjust(a,b,strength=1):
    norm_a = torch.linalg.norm(a)
    a = selfnorm(a)
    b = selfnorm(b)
    res = b - a * (a * b).sum()
    if res.isnan().any():
        res = torch.nan_to_num(res, nan=0.0)
    a = a - res * strength
    return a * norm_a

class condDiffSharpeningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "do_on": (["both","cond","uncond"], {"default": "both"},),
                                "scale": ("FLOAT",   {"default": 0.75, "min": -10.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                                "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, do_on, scale, start_at_sigma, end_at_sigma):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
        prev_cond   = None
        prev_uncond = None

        @torch.no_grad()
        def sharpen_conds_pre_cfg(args):
            nonlocal prev_cond, prev_uncond
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])

            sigma  = args["sigma"][0].item()
            first_step = sigma > (sigma_max - 1)
            if first_step:
                prev_cond   = None
                prev_uncond = None

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    if not first_step and sigma > end_at_sigma and sigma <= start_at_sigma:
                        if prev_cond is not None and do_on in ['both','cond']:
                            conds_out[0][b][c]   = normalize_adjust(conds_out[0][b][c], prev_cond[b][c], scale)
                        if prev_uncond is not None and uncond and do_on in ['both','uncond']:
                            conds_out[1][b][c] = normalize_adjust(conds_out[1][b][c], prev_uncond[b][c], scale)

            prev_cond = conds_out[0]
            if uncond:
                prev_uncond = conds_out[1]

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(sharpen_conds_pre_cfg)
        return (m, )

@torch.no_grad()
def normalized_pow(t,p):
    t_norm = t.norm()
    t_sign = t.sign()
    t_pow  = (t / t_norm).abs().pow(p)
    t_pow  = selfnorm(t_pow) * t_norm * t_sign
    return t_pow

class condExpNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "do_on": (["both","cond","uncond"], {"default": "both"},),
                                "exponent": ("FLOAT",   {"default": 0.8, "min": 0.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, do_on, exponent):
        @torch.no_grad()
        def exponentiate_conds_pre_cfg(args):
            if args["sigma"][0] <= 1: return args["conds_out"]

            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])

            for b in range(len(conds_out[0])):
                if do_on in ['both','cond']:
                    conds_out[0][b] = normalized_pow(conds_out[0][b], exponent)
                if uncond and do_on in ['both','uncond']:
                    conds_out[1][b] = normalized_pow(conds_out[1][b], exponent)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(exponentiate_conds_pre_cfg)
        return (m, )

@torch.no_grad()
def topk_average(latent, top_k=0.25, measure="average"):
    max_values = torch.topk(latent.flatten(), k=ceil(latent.numel()*top_k), largest=True ).values
    min_values = torch.topk(latent.flatten(), k=ceil(latent.numel()*top_k), largest=False).values
    value_range = measuring_methods[measure](max_values, min_values)
    return value_range

apply_scaling_methods = {
    "individual": lambda c, m, b: c * torch.tensor(m).view(c.shape[0],1,1).to(c.device),
    "all_as_one": lambda c, m, b: c * m[0],
    "average" :   lambda c, m, b: c * (sum(m) / len(m)),
    "smallest":   lambda c, m, b: c * min(m),
    "biggest" :   lambda c, m, b: c * max(m),
}

measuring_methods = {
    "difference": lambda x, y: (x.mean() - y.mean()).abs() / 2,
    "average":    lambda x, y: (x.mean() + y.abs().mean()) / 2,
    "biggest":    lambda x, y: max(x.mean(), y.abs().mean()),
}

class automatic_pre_cfg:
    @classmethod
    def INPUT_TYPES(s):
        scaling_methods_names = [k for k in apply_scaling_methods]
        return {"required": {
                                "model": ("MODEL",),
                                "scaling_method": (scaling_methods_names, {"default": scaling_methods_names[2]}),
                                "min_max_method": ([m for m in measuring_methods],),
                                # "top_k": ("FLOAT",   {"default": 0.25, "min": 0.01, "max": 0.5, "step": 1/100, "round": 1/100}),
                              },
                "optional": {
                                "channels_selection": ("CHANS",),
                }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scaling_method, min_max_method="difference", top_k=0.25, channels_selection=None):
        scaling_methods_names = [k for k in apply_scaling_methods]
        @torch.no_grad()
        def automatic_pre_cfg(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])

            if not uncond:
                return conds_out

            if channels_selection is None:
                channels = [True for _ in range(conds_out[0].shape[-3])]
            else:
                channels = channels_selection

            for b in range(len(conds_out[0])):
                chans = []
                if scaling_method == scaling_methods_names[1]:
                    if all(channels):
                        mes = topk_average(8 * conds_out[0][b] - 7 * conds_out[1][b], top_k=top_k, measure=min_max_method)
                    else:
                        cond_for_measure   = torch.stack([conds_out[0][b][j] for j in range(len(channels)) if channels[j]])
                        uncond_for_measure = torch.stack([conds_out[1][b][j] for j in range(len(channels)) if channels[j]])
                        mes = topk_average(8 * cond_for_measure - 7 * uncond_for_measure, top_k=top_k, measure=min_max_method)
                    chans.append(1 / max(mes,0.01))
                else:
                    for c in range(len(conds_out[0][b])):
                        if not channels[c]:
                            if scaling_method == scaling_methods_names[0]:
                                chans.append(1)
                            continue
                        mes = topk_average(8 * conds_out[0][b][c] - 7 * conds_out[1][b][c], top_k=top_k, measure=min_max_method)
                        new_scale = 1 / max(mes,0.01)
                        chans.append(new_scale)

                conds_out[0][b] = apply_scaling_methods[scaling_method](conds_out[0][b],chans,channels)
                conds_out[1][b] = apply_scaling_methods[scaling_method](conds_out[1][b],chans,channels)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(automatic_pre_cfg)
        return (m, )

class channel_selection_node:
    CHANNELS_AMOUNT = 4
    @classmethod
    def INPUT_TYPES(s):
        toggles = {f"channel_{x}" : ("BOOLEAN", {"default": True}) for x in range(s.CHANNELS_AMOUNT)}
        return {"required": toggles}

    RETURN_TYPES = ("CHANS",)
    FUNCTION = "exec"

    CATEGORY = "model_patches/Pre CFG/channels_selectors"

    def exec(self, **kwargs):
        chans = []
        for k, v in kwargs.items():
            if "channel_" in k:
                chans.append(v)
        return (chans, )

class individual_channel_selection_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "exclude" : ("BOOLEAN", {"default": False}),
                                "selected_channel": ("INT", {"default": 1, "min": 1, "max": 128}),
                                "total_channels"  : ("INT", {"default": 4, "min": 1, "max": 128}),
                              }
                            }

    RETURN_TYPES = ("CHANS",)
    FUNCTION = "exec"
    CATEGORY = "model_patches/Pre CFG/channels_selectors"
    def exec(self, exclude, selected_channel, total_channels):
        chans = [exclude for _ in range(total_channels)]
        chans[selected_channel - 1] = not exclude
        return (chans, )

class channel_multiplier_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "channel_1": ("FLOAT", {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "channel_2": ("FLOAT", {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "channel_3": ("FLOAT", {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "channel_4": ("FLOAT", {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "selection": (["both","cond","uncond"],),
                                "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, channel_1, channel_2, channel_3, channel_4, selection, start_at_sigma, end_at_sigma):
        chans = [channel_1, channel_2, channel_3, channel_4]
        @torch.no_grad()
        def channel_multiplier_function(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            sigma  = args["sigma"]
            if sigma[0] <= end_at_sigma or sigma[0] > start_at_sigma:
                return conds_out
            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    if selection in ["both","cond"]:
                        conds_out[0][b][c] *= chans[c]
                    if uncond and selection in ["both","uncond"]:
                        conds_out[1][b][c] *= chans[c]
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(channel_multiplier_function)
        return (m, )

class support_empty_uncond_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "method": (["divide by CFG","from cond"],),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, method):
        @torch.no_grad()
        def support_empty_uncond(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            cond_scale = args["cond_scale"]

            if not uncond and cond_scale > 1:
                if method == "divide by CFG":
                    conds_out[0] /= cond_scale
                else:
                    conds_out[1]  = conds_out[0]
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(support_empty_uncond)
        return (m, )

def replace_timestep(cond):
    cond = deepcopy(cond)
    cond[0]['timestep_start'] = 999999999.9
    cond[0]['timestep_end']   = 0.0
    return cond

def check_if_in_timerange(conds,timestep_in):
    for c in conds:
        all_good = True
        if 'timestep_start' in c:
            timestep_start = c['timestep_start']
            if timestep_in[0] > timestep_start:
                all_good = False
        if 'timestep_end' in c:
            timestep_end = c['timestep_end']
            if timestep_in[0] < timestep_end:
                all_good = False
        if all_good: return True
    return False

class zero_attention_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "do_on": (["cond","uncond"], {"default": "uncond"},),
                             "mix_scale":      ("FLOAT", {"default": 1.5,  "min": -2.0, "max": 2.0,    "step": 1/2,   "round": 1/100}),
                             "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                             "end_at_sigma":   ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                            #  "attention":      (["both","self","cross"],),
                            #  "unet_block":     (["input","middle","output"],),
                            #  "unet_block_id":  ("INT", {"default": 8, "min": 0, "max": 20}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, do_on, mix_scale, start_at_sigma, end_at_sigma, attention="both", unet_block="input", unet_block_id=8):
        cond_index = 1 if do_on == "uncond" else 0
        attn = {"both":["attn1","attn2"],"self":["attn1"],"cross":["attn2"]}[attention]

        def zero_attention_function(q, k, v, extra_options, mask=None):
            return torch.zeros_like(q)

        @torch.no_grad()
        def zero_attention_pre_cfg_patch(args):
            conds_out = args["conds_out"]
            sigma = args["sigma"][0].item()

            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return conds_out

            conds = args["conds"]
            cond_to_process = conds[cond_index]
            cond_generated  = torch.any(conds_out[cond_index])

            if not cond_generated:
                cond_to_process = replace_timestep(cond_to_process)
            elif mix_scale == 1:
                print(" Mix scale at one!\nPrediction generated for nothing.\nUse the node ConditioningSetTimestepRange to avoid generating if you want to use the full result.")

            model_options = deepcopy(args["model_options"])
            for att in attn:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, zero_attention_function, att, unet_block, unet_block_id)

            (noise_pred,) = calc_cond_batch(args['model'], [cond_to_process], args['input'], args['timestep'], model_options)

            if mix_scale == 1 or not cond_generated:
                conds_out[cond_index] = noise_pred
            elif cond_generated:
                conds_out[cond_index] = weighted_average(noise_pred,conds_out[cond_index],mix_scale)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(zero_attention_pre_cfg_patch)
        return (m, )

class perturbed_attention_guidance_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "scale":          ("FLOAT", {"default": 0.5,  "min": -2.0, "max": 10.0,   "step": 1/20,  "round": 1/100}),
                             "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                             "end_at_sigma":   ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scale, start_at_sigma, end_at_sigma, do_on="cond", attention="self", unet_block="middle", unet_block_id=0):
        cond_index = 1 if do_on == "uncond" else 0
        attn = {"both":["attn1","attn2"],"self":["attn1"],"cross":["attn2"]}[attention]

        def perturbed_attention_guidance(q, k, v, extra_options, mask=None):
            return v

        @torch.no_grad()
        def perturbed_attention_guidance_pre_cfg_patch(args):
            conds_out = args["conds_out"]
            sigma = args["sigma"][0].item()

            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return conds_out

            conds = args["conds"]
            cond_to_process = conds[cond_index]
            cond_generated  = torch.any(conds_out[cond_index])

            if not cond_generated:
                return conds_out

            model_options = deepcopy(args["model_options"])
            for att in attn:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, perturbed_attention_guidance, att, unet_block, unet_block_id)

            (noise_pred,) = calc_cond_batch(args['model'], [cond_to_process], args['input'], args['timestep'], model_options)

            conds_out[cond_index] = conds_out[cond_index] + (conds_out[cond_index] - noise_pred) * scale

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(perturbed_attention_guidance_pre_cfg_patch)
        return (m, )

@torch.no_grad()
def euclidean_weights(tensors,exponent=2,proximity_exponent=1,min_score_into_zeros=0):
    divider = tensors.shape[0]
    if exponent == 0:
        exponent = 6.55
    device = tensors.device
    distance_weights = torch.zeros_like(tensors).to(device = device)

    for i in range(len(tensors)):
        for j in range(len(tensors)):
            if i == j: continue
            current_distance = (tensors[i] - tensors[j]).abs() / divider
            if proximity_exponent > 1:
                current_distance = current_distance ** proximity_exponent
            distance_weights[i] += current_distance

    min_stack, _ = torch.min(distance_weights, dim=0)
    max_stack, _ = torch.max(distance_weights, dim=0)
    max_stack    = torch.where(max_stack == 0, torch.tensor(1), max_stack)
    sum_of_weights = torch.zeros_like(tensors[0]).to(device = device)

    max_stack -= min_stack

    for i in range(len(tensors)):
        distance_weights[i] -= min_stack
        distance_weights[i] /= max_stack
        distance_weights[i]  = 1 - distance_weights[i]
        distance_weights[i]  = torch.clamp(distance_weights[i], min=0)
        if min_score_into_zeros > 0:
            distance_weights[i]  = torch.where(distance_weights[i] < min_score_into_zeros, torch.zeros_like(distance_weights[i]), distance_weights[i])
        
        if exponent > 1:
            distance_weights[i] = distance_weights[i] ** exponent

        sum_of_weights += distance_weights[i]
    
    mean_score = (sum_of_weights.mean() / divider) ** exponent

    sum_of_weights = torch.where(sum_of_weights == 0, torch.zeros_like(sum_of_weights) + 1 / divider, sum_of_weights)
    result = torch.zeros_like(tensors[0]).to(device = device)

    for i in range(len(tensors)):
        distance_weights[i] /= sum_of_weights
        distance_weights[i]  = torch.where(torch.isnan(distance_weights[i]) | torch.isinf(distance_weights[i]), torch.zeros_like(distance_weights[i]), distance_weights[i])
        result = result + tensors[i] * distance_weights[i]

    return result, mean_score

class condConsensusSharpeningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "scale": ("FLOAT",   {"default": 0.75, "min": -10.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                                "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scale, start_at_sigma, end_at_sigma):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
        prev_conds   = []
        prev_unconds = []

        @torch.no_grad()
        def sharpen_conds_pre_cfg(args):
            nonlocal prev_conds, prev_unconds
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            
            sigma = args["sigma"][0].item()

            if sigma <= end_at_sigma:
                return conds_out

            first_step = sigma > (sigma_max - 1)
            if first_step:
                prev_conds   = []
                prev_unconds = []

            prev_conds.append(conds_out[0] / conds_out[0].norm())
            if uncond:
                prev_unconds.append(conds_out[1] / conds_out[1].norm())

            if sigma > start_at_sigma:
                return conds_out

            if len(prev_conds) > 3:
                consensus_cond, mean_score_cond = euclidean_weights(torch.stack(prev_conds))
                consensus_cond = consensus_cond * conds_out[0].norm()
            if len(prev_unconds) > 3 and uncond:
                consensus_uncond, mean_score_uncond = euclidean_weights(torch.stack(prev_unconds))
                consensus_uncond = consensus_uncond * conds_out[1].norm()

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                        if len(prev_conds) > 3:
                            conds_out[0][b][c] = normalize_adjust(conds_out[0][b][c], consensus_cond[b][c],  mean_score_cond * scale)
                        if len(prev_unconds) > 3 and uncond:
                            conds_out[1][b][c] = normalize_adjust(conds_out[1][b][c], consensus_uncond[b][c], mean_score_uncond * scale)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(sharpen_conds_pre_cfg)
        return (m, )

def sigma_to_percent(model_sampling, sigma_value):
    if sigma_value >= 999999999.9:
        return 0.0
    if sigma_value <= 0.0:
        return 1.0
    sigma_tensor = torch.tensor([sigma_value], dtype=torch.float32)
    timestep = model_sampling.timestep(sigma_tensor)
    percent = 1.0 - (timestep.item() / 999.0)
    return percent

class ConditioningSetTimestepRangeFromSigma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "conditioning": ("CONDITIONING", ),
                             "sigma_start" : ("FLOAT", {"default": 15.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                             "sigma_end"   : ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 10000.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_range"

    CATEGORY = "advanced/conditioning"

    def set_range(self, model, conditioning, sigma_start, sigma_end):
        model_sampling = model.model.model_sampling
        (c, ) = ConditioningSetTimestepRange.set_range(conditioning,sigma_to_percent(model_sampling, sigma_start),sigma_to_percent(model_sampling, sigma_end))
        return (c, )

class ShapeAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 1.5,  "min": 0.0, "max": 10.0, "step": 1/10, "round": 1/100}),
                # "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                # "end_at_sigma":   ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                # "enabled" :  ("BOOLEAN", {"default": True}),
                # "attention":      (["both","self","cross"],),
                # "unet_block":     (["input","middle","output"],),
                # "unet_block_id":  ("INT", {"default": 8, "min": 0, "max": 20}), # uncomment these lines if you want to have fun with the other layers
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, scale, start_at_sigma=999999999.9, end_at_sigma=0.0, enabled=True, attention="both", unet_block="input", unet_block_id=8):
        attn = {"both":["attn1","attn2"],"self":["attn1"],"cross":["attn2"]}[attention]
        if scale == 1:
            print(" Shape attention disabled (scale is one)")
        if not enabled or scale == 1:
            return (model,)

        m = model.clone()

        def shape_attention(q, k, v, extra_options, mask=None):
            sigma = extra_options['sigmas'][0]
            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return default_attention(q, k, v, extra_options['n_heads'], mask)
            if scale != 0:
                return default_attention(q, k, v, extra_options['n_heads'], mask) * scale
            else:
                return torch.zeros_like(q)

        for att in attn:
            m.model_options = comfy.model_patcher.set_model_options_patch_replace(m.model_options, shape_attention, att, unet_block, unet_block_id)

        return (m,)

class PreCFGsubtractMeanNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                # "per_channel" : ("BOOLEAN", {"default": False}),
                "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                "end_at_sigma":   ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                "enabled" : ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, start_at_sigma, end_at_sigma, enabled, per_channel=False):
        if not enabled: return (model,)
        m = model.clone()
        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            sigma = args["sigma"][0].item()
            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return conds_out
            for x in range(len(conds_out)):
                if torch.any(conds_out[x]):
                    for b in range(len(conds_out[x])):
                        if per_channel:
                            for c in range(len(conds_out[x][b])):
                                conds_out[x][b][c] -= conds_out[x][b][c].mean()
                        else:
                            conds_out[x][b] -= conds_out[x][b].mean()
            return conds_out
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m,)

class PostCFGsubtractMeanNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                # "per_channel" : ("BOOLEAN", {"default": False}),
                "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                "end_at_sigma":   ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                "enabled" : ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, start_at_sigma, end_at_sigma, enabled, per_channel=False):
        if not enabled: return (model,)
        m = model.clone()
        def post_cfg_function(args):
            cfg_result = args["denoised"]
            sigma = args["sigma"][0].item()
            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return cfg_result
            for b in range(len(cfg_result)):
                if per_channel:
                    for c in range(len(cfg_result[b])):
                        cfg_result[b][c] -= cfg_result[b][c].mean()
                else:
                    cfg_result[b] -= cfg_result[b].mean()
            return cfg_result
        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return (m,)

class PostCFGDotNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "batch":   ("INT", {"default": 0,  "min": 0, "max": 100,  "step": 1}),
                "channel": ("INT", {"default": 0,  "min": 0, "max": 100,  "step": 1}),
                "coord_x": ("INT", {"default": 64, "min": 0, "max": 1000, "step": 1}),
                "coord_y": ("INT", {"default": 64, "min": 0, "max": 1000, "step": 1}),
                "value":   ("FLOAT", {"default": 1, "min": -10.0, "max": 10.0, "step": 1/10, "round": 1/100}),
                "start_at_sigma": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1000.0, "step": 1/100, "round": 1/100}),
                "end_at_sigma":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 1/100, "round": 1/100}),
                "enabled" : ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, batch, channel, coord_x, coord_y, value, start_at_sigma, end_at_sigma, enabled):
        if not enabled: return (model,)
        m = model.clone()
        def post_cfg_function(args):
            cfg_result = args["denoised"]
            sigma = args["sigma"][0].item()
            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return cfg_result

            channel_norm = cfg_result[batch][channel].norm()
            cfg_result[batch][channel] /= channel_norm
            cfg_result[batch][channel][coord_y][coord_x] = value
            cfg_result[batch][channel] *= channel_norm

            return cfg_result

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return (m,)

class uncondZeroPreCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "scale":     ("FLOAT",   {"default": 0.75, "min": 0.0, "max": 10.0, "step": 1/20, "round": 0.01}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scale):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()

        @torch.no_grad()
        def uncond_zero_pre_cfg(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            sigma  = args["sigma"][0].item()
            if uncond or sigma < (sigma_max * 0.069):
                return conds_out

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    mes = topk_average(conds_out[0][b][c], measure="difference") ** 0.5
                    conds_out[0][b][c] = conds_out[0][b][c] * scale / mes
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(uncond_zero_pre_cfg)
        return (m, )

class latent_color_control_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "Red":   ("FLOAT", {"default": 0, "min": -2.0, "max": 2.0, "step": 1/100, "round": 1/100}),
                                "Green": ("FLOAT", {"default": 0, "min": -2.0, "max": 2.0, "step": 1/100, "round": 1/100}),
                                "Blue":  ("FLOAT", {"default": 0, "min": -2.0, "max": 2.0, "step": 1/100, "round": 1/100}),
                                "selection": (["both","cond","uncond"],),
                                "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, Red, Green, Blue, selection, start_at_sigma, end_at_sigma):
        latent_rgb_factors = sdxl_latent_rgb_factors
        rgb = [Red, Green, Blue]
        @torch.no_grad()
        def latent_control_pre_cfg_function(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            sigma  = args["sigma"]

            if sigma[0] <= end_at_sigma or sigma[0] > start_at_sigma or all(c == 0 for c in rgb):
                return conds_out

            conds_index = []
            if selection in ["both","cond"]:
                conds_index.append(0)
            if uncond and selection in ["both","uncond"]:
                conds_index.append(1)

            for i in conds_index:
                for b in range(len(conds_out[i])):
                    cond_norm  = conds_out[i][b].norm()
                    color_cond = torch.zeros_like(conds_out[i][b])
                    for c in range(len(conds_out[i][b])):
                        for r in range(len(rgb)):
                            if rgb[r] != 0:
                                color_cond[c] += rgb[r] * latent_rgb_factors[c][r] / 0.13025
                    conds_out[i][b] = selfnorm(conds_out[i][b] + color_cond) * cond_norm
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(latent_control_pre_cfg_function)
        return (m, )

class variable_scale_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "start_multiplier": ("FLOAT", {"default": 1.5,  "min": 0.0, "max": 10.0,  "step": 1/100, "round": 1/100}),
                             "end_multiplier":   ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,  "step": 1/100, "round": 1/100}),
                             "proportional_to": (["sigma","steps progression"],),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, start_multiplier, end_multiplier, proportional_to):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()

        @torch.no_grad()
        def variable_scale_pre_cfg_patch(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])

            if not uncond:
                return conds_out

            sigma = args["sigma"][0].item()
            if proportional_to == "steps progression":
                progression = sigma_to_percent(model_sampling, sigma)
            else:
                progression = 1 - sigma / sigma_max

            progression = max(min(progression, 1), 0)
            current_multiplier = start_multiplier * (1 - progression) + end_multiplier * progression

            conds_out[0] = conds_out[0] * current_multiplier
            conds_out[1] = conds_out[1] * current_multiplier

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(variable_scale_pre_cfg_patch)
        return (m, )
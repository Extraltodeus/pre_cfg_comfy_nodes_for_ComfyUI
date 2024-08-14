import torch
import torch.nn.functional as F
from math import ceil, floor
from copy import deepcopy
import comfy.model_patcher
from comfy.sampler_helpers import convert_cond
from comfy.samplers import calc_cond_batch, encode_model_conds
from comfy.ldm.modules.attention import optimized_attention_for_device
from nodes import ConditioningConcat, ConditioningSetTimestepRange
import comfy.model_management as model_management
from comfy.latent_formats import SDXL as SDXL_Latent
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
SDXL_Latent = SDXL_Latent()
sdxl_latent_rgb_factors = SDXL_Latent.latent_rgb_factors
ConditioningConcat = ConditioningConcat()
ConditioningSetTimestepRange = ConditioningSetTimestepRange()
default_attention = optimized_attention_for_device(model_management.get_torch_device())
default_device = model_management.get_torch_device()

weighted_average = lambda tensor1, tensor2, weight1: (weight1 * tensor1 + (1 - weight1) * tensor2)
selfnorm = lambda x: x / x.norm()
minmaxnorm = lambda x: torch.nan_to_num((x - x.min()) / (x.max() - x.min()), nan=0.0, posinf=1.0, neginf=0.0)
normlike = lambda x, y: x / x.norm() * y.norm()

def get_sigma_min_max(model):
    model_sampling = model.model.model_sampling
    sigma_min = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_min)).item()
    sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
    return sigma_min, sigma_max

@torch.no_grad()
def make_new_uncond_at_scale(cond,uncond,cond_scale,new_scale):
    new_scale_ratio = (new_scale - 1) / (cond_scale - 1)
    return cond * (1 - new_scale_ratio) + uncond * new_scale_ratio

@torch.no_grad()
def make_new_uncond_at_scale_co(conds_out,cond_scale,new_scale):
    new_scale_ratio = (new_scale - 1) / (cond_scale - 1)
    return conds_out[0] * (1 - new_scale_ratio) + conds_out[1] * new_scale_ratio

@torch.no_grad()
def get_denoised_at_scale(x_orig,cond,uncond,cond_scale):
    return x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))

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
                                # "cond_or_uncond":  (["both","uncond"], {"default":"uncond"}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, clip, neg_scale, set_context_length, context_length, start_at_sigma, end_at_sigma, cond_or_uncond="uncond"):
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

            if cond_or_uncond == "both":
                perp_p = pos - ((torch.mul(neg, pos).sum())/(torch.norm(neg)**2)) * neg
                perp_pos = perp_p * neg_scale
                conds_out[0] = noise_pred_nocond + perp_pos
            else:
                conds_out[0] = noise_pred_nocond + pos            
            conds_out[1] = noise_pred_nocond + perp_neg

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_perp_neg_function)
        return (m, )

class pre_cfg_re_negative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "empty_proportion": ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,  "step": 1/20, "round": 0.01}),
                                "progressive_scale" : ("BOOLEAN", {"default": False}),
                                "set_context_length" : ("BOOLEAN", {"default": False}),
                                "context_length": ("INT", {"default": 1,  "min": 1, "max": 100,  "step": 1}),
                                "end_at_sigma":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, clip, empty_proportion, progressive_scale, set_context_length, context_length, end_at_sigma):
        sigma_min, sigma_max = get_sigma_min_max(model)
        empty_cond, pooled = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        nocond = [[empty_cond, {"pooled_output": pooled}]]
        if context_length > 1 and set_context_length:
            short_nocond = deepcopy(nocond)
            for x in range(context_length - 1):
                (nocond,) = ConditioningConcat.concat(nocond, short_nocond)
        nocond = convert_cond(nocond)

        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            sigma = args["sigma"][0]
            # cond_scale = args["cond_scale"]

            if sigma <= end_at_sigma or not torch.any(conds_out[1]):
                return conds_out

            model_options = args["model_options"]
            timestep = args["timestep"]
            model  = args["model"]
            x_orig = args["input"]

            nocond_processed = encode_model_conds(model.extra_conds, nocond, x_orig, x_orig.device, "negative")
            (noise_pred_nocond,) = calc_cond_batch(model, [nocond_processed], x_orig, timestep, model_options)
            if progressive_scale:
                progression   = (sigma - sigma_min) / (sigma_max - sigma_min)
                current_scale = progression * empty_proportion + (1 - progression) * (1 - empty_proportion)
                current_scale = torch.clamp(current_scale, min=0, max=1)
                conds_out[1]  = current_scale * noise_pred_nocond + conds_out[1] * (1 - current_scale)
            else:
                conds_out[1] = empty_proportion * noise_pred_nocond + conds_out[1] * (1 - empty_proportion)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
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

            if do_on in ['both','uncond'] and not uncond:
                return conds_out

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
    "individual": lambda c, m: c * torch.tensor(m).view(c.shape[0],1,1).to(c.device),
    "all_as_one": lambda c, m: c * m[0],
    "average_of_all_channels" :   lambda c, m: c * (sum(m) / len(m)),
    "smallest_of_all_channels":   lambda c, m: c * min(m),
    "biggest_of_all_channels" :   lambda c, m: c * max(m),
}

measuring_methods = {
    "difference": lambda x, y: (x.mean() - y.mean()).abs() / 2,
    "average":    lambda x, y: (x.mean() + y.abs().mean()) / 2,
    "biggest":    lambda x, y: max(x.mean(), y.abs().mean()),
}

class automatic_pre_cfg:
    @classmethod
    def INPUT_TYPES(s):
        scaling_methods_names   = [k for k in apply_scaling_methods]
        measuring_methods_names = [k for k in measuring_methods]
        return {"required": {
                                "model": ("MODEL",),
                                "scaling_method": (scaling_methods_names, {"default": scaling_methods_names[0]}),
                                "min_max_method": ([m for m in measuring_methods], {"default": measuring_methods_names[1]}),
                                "reference_CFG":  ("FLOAT", {"default": 8, "min": 0.0, "max": 100, "step": 1/10, "round": 1/100}),
                                "scale_multiplier": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 100, "step": 1/100, "round": 1/100}),
                                "top_k": ("FLOAT",   {"default": 0.25, "min": 0.0, "max": 0.5, "step": 1/20, "round": 1/100}),
                              },
                "optional": {
                                "channels_selection": ("CHANS",),
                }
                              }
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("MODEL","parameters",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scaling_method, min_max_method="difference", reference_CFG=8, scale_multiplier=0.8, top_k=0.25, channels_selection=None):
        parameters_string = f"scaling_method: {scaling_method}\nmin_max_method: {min_max_method}"
        if channels_selection is not None:
            for x in range(channels_selection):
                parameters_string += f"\nchannel {x+1}: {channels_selection[x]}"
        scaling_methods_names = [k for k in apply_scaling_methods]
        @torch.no_grad()
        def automatic_pre_cfg(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            uncond = torch.any(conds_out[1])
            if reference_CFG == 0:
                reference_scale = cond_scale
            else:
                reference_scale = reference_CFG

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
                        mes = topk_average(reference_scale * conds_out[0][b] - (reference_scale - 1) * conds_out[1][b], top_k=top_k, measure=min_max_method)
                    else:
                        cond_for_measure   = torch.stack([conds_out[0][b][j] for j in range(len(channels)) if channels[j]])
                        uncond_for_measure = torch.stack([conds_out[1][b][j] for j in range(len(channels)) if channels[j]])
                        mes = topk_average(reference_scale * cond_for_measure - (reference_scale - 1) * uncond_for_measure, top_k=top_k, measure=min_max_method)
                    chans.append(scale_multiplier / max(mes,0.01))
                else:
                    for c in range(len(conds_out[0][b])):
                        if not channels[c]:
                            if scaling_method == scaling_methods_names[0]:
                                chans.append(1)
                            continue
                        mes = topk_average(reference_scale * conds_out[0][b][c] - (reference_scale - 1) * conds_out[1][b][c], top_k=top_k, measure=min_max_method)
                        new_scale = scale_multiplier / max(mes,0.01)
                        chans.append(new_scale)


                conds_out[0][b] = apply_scaling_methods[scaling_method](conds_out[0][b],chans)
                conds_out[1][b] = apply_scaling_methods[scaling_method](conds_out[1][b],chans)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(automatic_pre_cfg)
        return (m, parameters_string,)

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
                             "method": (["from cond","divide by CFG"],),
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
                    conds_out[1]  = conds_out[0].clone()
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
                print(" Mix scale at one!\nPrediction not generated.\nUse the node ConditioningSetTimestepRange to avoid generating if you want to use this node.")
                return conds_out

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

    def patch(self, model, scale, start_at_sigma=999999999.9, end_at_sigma=0.0, enabled=True, attention="self", unet_block="input", unet_block_id=8):
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

class ExlAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 2,  "min": -1.0, "max": 10.0, "step": 1/10, "round": 1/100}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, scale, enabled):
        if not enabled:
            return (model,)
        m = model.clone()
        def cross_patch(q, k, v, extra_options, mask=None):
            first_attention  = default_attention(q, k, v, extra_options['n_heads'], mask)
            second_attention = normlike(q+(q-default_attention(first_attention, k, v, extra_options['n_heads'])), first_attention) * scale
            return second_attention
        m.model_options = comfy.model_patcher.set_model_options_patch_replace(m.model_options, cross_patch, "attn2", "middle", 0)
        return (m,)

class PreCFGsubtractMeanNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                # "per_channel" : ("BOOLEAN", {"default": False}), #It's just not good
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
                # "per_channel" : ("BOOLEAN", {"default": False}), #It's just not good
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
        scaling_methods_names = [k for k in apply_scaling_methods]
        return {"required": {
                                "model": ("MODEL",),
                                "scale": ("FLOAT",   {"default": 0.75, "min": 0.0, "max": 10.0, "step": 1/20, "round": 0.01}),
                                "start_at_sigma": ("FLOAT", {"default": 100, "min": 0.0, "max": 1000.0, "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 1/100, "round": 1/100}),
                                "scaling_method": (scaling_methods_names, {"default": scaling_methods_names[2]}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scale, start_at_sigma, end_at_sigma, scaling_method):
        scaling_methods_names = [k for k in apply_scaling_methods]
        @torch.no_grad()
        def uncond_zero_pre_cfg(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            sigma  = args["sigma"][0].item()
            if uncond or sigma <= end_at_sigma or sigma > start_at_sigma:
                return conds_out

            for b in range(len(conds_out[0])):
                chans = []
                if scaling_method == scaling_methods_names[1]:
                    mes = topk_average(8 * conds_out[0][b] - 7 * conds_out[1][b], measure="difference")
                for c in range(len(conds_out[0][b])):
                    mes = topk_average(conds_out[0][b][c], measure="difference") ** 0.5
                    chans.append(scale / mes)
                conds_out[0][b] = apply_scaling_methods[scaling_method](conds_out[0][b],chans)
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(uncond_zero_pre_cfg)
        return (m, )

class variable_scale_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "target_scale": ("FLOAT", {"default": 5.0,  "min": 1.0, "max": 100.0,  "step": 1/2, "round": 1/100}),
                             "target_as_start": ("BOOLEAN", {"default": True}),
                             "proportional_to": (["sigma","steps progression"],),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, target_scale, target_as_start, proportional_to):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()

        @torch.no_grad()
        def variable_scale_pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            sigma  = args["sigma"][0].item()
            scales = [cond_scale,target_scale]

            if not torch.any(conds_out[1]):
                return conds_out

            if proportional_to == "steps progression":
                progression = sigma_to_percent(model_sampling, sigma)
            else:
                progression = 1 - sigma / sigma_max
            progression = max(min(progression, 1), 0)

            current_scale = scales[target_as_start] * (1 - progression) + scales[not target_as_start] * progression
            new_scale = (current_scale - 1) / (cond_scale - 1)
            conds_out[1] = weighted_average(conds_out[1], conds_out[0], new_scale)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(variable_scale_pre_cfg_patch)
        return (m, )

class latent_noise_subtract_mean_node:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "latent_input": ("LATENT", {"forceInput": True}),
                "enabled" : ("BOOLEAN", {"default": True}),
                }}
    FUNCTION = "exec"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent"

    def exec(self, latent_input, enabled):
        if not enabled:
            return (latent_input,)
        new_latents = deepcopy(latent_input)
        for x in range(len(new_latents['samples'])):
                new_latents['samples'][x] -= torch.mean(new_latents['samples'][x])
        return (new_latents,)

class flip_flip_conds_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "enabled" : ("BOOLEAN", {"default": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, enabled):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            uncond = torch.any(conds_out[1])

            if not uncond or not enabled:
                return conds_out

            conds_out[0], conds_out[1] = conds_out[1], conds_out[0]
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class norm_uncond_to_cond_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "enabled" : ("BOOLEAN", {"default": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, enabled):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            uncond = torch.any(conds_out[1])

            if not uncond or not enabled:
                return conds_out

            conds_out[1] = conds_out[1] / conds_out[1].norm() * conds_out[0].norm()
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class replace_uncond_channel_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "channel": ("INT", {"default": 1,  "min": 1, "max": 128,  "step": 1}),
                             "enabled" : ("BOOLEAN", {"default": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, channel, enabled):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            uncond = torch.any(conds_out[1])

            if not uncond or not enabled:
                return conds_out

            for b in range(len(conds_out[0])):
                if len(conds_out[1][b]) < channel:
                    print(F" WRONG CHANNEL SELECTED. THE LATENT SPACE ONLY HAS {len(conds_out[1][b])} CHANNELS")
                else:
                    conds_out[1][b][channel - 1] = conds_out[0][b][channel - 1]

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class merge_uncond_channel_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "channel": ("INT", {"default": 1,  "min": 1, "max": 128,  "step": 1}),
                             "CFG_scale": ("FLOAT", {"default": 5, "min": 2.0, "max": 100.0, "step": 1/2, "round": 1/100}),
                             "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                             "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                             "enabled" : ("BOOLEAN", {"default": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, channel, CFG_scale, start_at_sigma, end_at_sigma, enabled):
        if not enabled: return model,
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"][0].item()

            if not torch.any(conds_out[1]) or sigma <= end_at_sigma or sigma > start_at_sigma:
                return conds_out

            for b in range(len(conds_out[0])):
                if len(conds_out[1][b]) < channel:
                    print(F" WRONG CHANNEL SELECTED. THE LATENT SPACE ONLY HAS {len(conds_out[1][b])} CHANNELS")
                else:
                    new_scale = (CFG_scale - 1) / (cond_scale - 1)
                    conds_out[1][b][channel - 1] = weighted_average(conds_out[1][b][channel - 1], conds_out[0][b][channel - 1], new_scale)
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class multiply_cond_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "selection": (["both","cond","uncond"],),
                             "value":  ("FLOAT", {"default": 0, "min": -100.0, "max": 100.0, "step": 1/100, "round": 1/100}),
                             "enabled" : ("BOOLEAN", {"default": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, selection, value, enabled):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            uncond = torch.any(conds_out[1])

            if (not uncond and selection in ["both","uncond"]) or not enabled:
                return conds_out

            if selection in ["both","cond"]:
                conds_out[0] = conds_out[0] * value
            if selection in ["both","uncond"]:
                conds_out[1] = conds_out[1] * value

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

def generate_gradient_mask(tensor, horizontal=False):
    dim = 3 if horizontal else 2
    gradient = torch.linspace(0, 1, steps=tensor.size(dim), device=tensor.device)
    if horizontal:
        merging_gradient = gradient.repeat(tensor.size(0), tensor.size(1), tensor.size(2), 1)
    else:
        merging_gradient = gradient.unsqueeze(1).repeat(tensor.size(0), tensor.size(1), 1, tensor.size(3))
    return merging_gradient

class gradient_scaling_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "maximum_scale": ("FLOAT", {"default": 80,  "min": 0.0, "max": 1000.0, "step": 1, "round": 1/100, "tooltip":"It is an equivalent to the CFG scale."}),
                             "minimum_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 10.0,   "step": 1/2, "round": 1/100, "tooltip":"It is an equivalent to the CFG scale."}),
                             "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 1/10, "round": 1/10}),
                             "end_at_sigma": ("FLOAT", {"default": 0.28,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                            #  "free_scale" : ("BOOLEAN", {"default": False}),
                             "converging_scales" : ("BOOLEAN", {"default": True}),
                            #  "noise_add_diff" : ("BOOLEAN", {"default": True}),
                            #  "split_channels" : ("BOOLEAN", {"default": False}),
                             "invert_mask" : ("BOOLEAN", {"default": False}),
                            #  "no_input" : (["rand","rev","cond","uncond","swap","r_swap","diff","add_diff","rand_rev","rev_cond","rand_cond","rev_cond_sp","cond_rev_sp"],),
                            #  "start_at_sigma": ("FLOAT", {"default": 15,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                            #  "end_at_sigma": ("FLOAT", {"default": 0.28,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                             },
                             "optional":{
                                 "input_mask": ("MASK", {"tooltip":"If only a mask is connected the scale becomes a CFG scale of what is being masked.\nWhen a latent is connected the mask defines what will be modified by the node."},),
                                 "input_latent": ("LATENT", {"tooltip":"If a latent is connected the scale becomes the maximum scale allowed in which to seek similarity."},),
                                }
                             }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def get_latent_guidance_mask_channel(self,x_orig,cond,uncond,guide,minimum_scale,maximum_scale,noise_add_diff):
        scales = torch.zeros_like(x_orig, device=x_orig.device)
        for b in range(cond.shape[0]):
            for c in range(cond.shape[1]):
                scales[b][c] = self.get_latent_guidance_mask(x_orig[b][c],cond[b][c],uncond[b][c],guide[0][c],minimum_scale,maximum_scale,noise_add_diff)
        return scales

    @torch.no_grad()
    def get_latent_guidance_mask(self,x_orig,cond,uncond,guide,minimum_scale,maximum_scale,noise_add_diff):
        low_denoised  = get_denoised_at_scale(x_orig,cond,uncond,minimum_scale)
        high_denoised = get_denoised_at_scale(x_orig,cond,uncond,maximum_scale)
        if noise_add_diff:
            guide = guide + (guide - (x_orig * guide.norm() / x_orig.norm()))
        guide = guide / guide.norm()
        low_diff  = (low_denoised  - guide * low_denoised.norm()).abs()
        high_diff = (high_denoised - guide * high_denoised.norm()).abs()
        return torch.clamp(low_diff / high_diff, min=0, max=1)

    def patch(self, model, maximum_scale, minimum_scale, invert_mask, strength, end_at_sigma, start_at_sigma=99999, no_input="swap", noise_add_diff=True, converging_scales=False, split_channels=False, free_scale=False, input_mask=None, input_latent=None):
        sigma_min, sigma_max = get_sigma_min_max(model)
        model_sampling = model.model.model_sampling
        scaling_function = self.get_latent_guidance_mask_channel if split_channels else self.get_latent_guidance_mask
        mask_as_weight = None
        latent_as_guidance = None
        random_guidance = False
        if input_mask is not None:
            mask_as_weight = input_mask.clone().to(device=default_device)
            if invert_mask:
                mask_as_weight = 1 - mask_as_weight
            if mask_as_weight.dim() == 3:
                mask_as_weight = mask_as_weight.unsqueeze(1)
        if input_latent is not None:
            latent_as_guidance = input_latent["samples"].clone().to(device=default_device)
        elif input_mask is None:
            random_guidance = True

        snc = lambda x: x / x.norm()
        trl = lambda x: torch.randn_like(x,device=x.device)
        no_input_operations = {
            "rand": lambda x, y, o, z, s: snc(trl(x)) * x.norm(),
            "rev": lambda x, y, o, z, s: x * -1,
            "cond": lambda x, y, o, z, s: snc(y) * x.norm(),
            "uncond": lambda x, y, o, z, s: snc(o) * x.norm() * -1,
            "swap": lambda x, y, o, z, s: no_input_operations["cond"](x, y, o, z, s) if s > 0.36 else no_input_operations["uncond"](x, y, o, z, s),
            "r_swap": lambda x, y, o, z, s: no_input_operations["cond"](x, y, o, z, s) if s <= 0.36 else no_input_operations["uncond"](x, y, o, z, s),
            "diff": lambda x, y, o, z, s: snc(y - o) * x.norm(),
            "add_diff": lambda x, y, o, z, s: snc(y + y - o) * x.norm(),
            "rand_rev": lambda x, y, o, z, s: snc(trl(x)) * x.norm(),
            "rev_cond": lambda x, y, o, z, s: (snc(x) * -1 + snc(y) * 0.5) * x.norm() / 1.5,
            "rand_cond": lambda x, y, o, z, s: (snc(x) * -1 + snc(trl(x)) * 0.5) * x.norm() / 1.5,
            "rev_cond_sp": lambda x, y, o, z, s: no_input_operations["rev"](x,y,z) * z + (1 - z) * no_input_operations["cond"](x,y,z),
            "cond_rev_sp": lambda x, y, o, z, s: no_input_operations["rev"](x,y,z) * (1 - z) + z * no_input_operations["cond"](x,y,z),
        }

        @torch.no_grad()
        def pre_cfg_patch(args):
            nonlocal mask_as_weight, latent_as_guidance
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args['input']
            sigma  = args["sigma"][0]
            sp = min(1,max(0,sigma_to_percent(model_sampling, sigma - sigma_min * 3) + 1 / 100)) ** 2

            if not torch.any(conds_out[1]) or sigma <= end_at_sigma or sigma > start_at_sigma or (converging_scales and sp == 1):
                return conds_out

            if converging_scales:
                current_maximum_scale = sp * cond_scale + (1 - sp) * maximum_scale
                current_minimum_scale = sp * cond_scale + (1 - sp) * minimum_scale
            else:
                current_maximum_scale = maximum_scale
                current_minimum_scale = minimum_scale

            if mask_as_weight is not None and mask_as_weight.shape[-2:] != conds_out[1].shape[-2:]:
                mask_as_weight = F.interpolate(mask_as_weight, size=(conds_out[1].shape[-2], conds_out[1].shape[-1]), mode='bilinear', align_corners=False)
            
            if random_guidance:
                latent_as_guidance = no_input_operations[no_input](x_orig.clone(),conds_out[0].clone(),conds_out[1].clone(),sp,sigma/sigma_max)

            if latent_as_guidance is not None:
                if latent_as_guidance.shape[-2:] != conds_out[1].shape[-2:]:
                    latent_as_guidance = F.interpolate(latent_as_guidance, size=(conds_out[1].shape[-2], conds_out[1].shape[-1]), mode='bilinear', align_corners=False)

                scaling_weight = scaling_function(x_orig,conds_out[0],conds_out[1],latent_as_guidance.clone(),current_minimum_scale,current_maximum_scale,noise_add_diff)

                target_scales = scaling_weight * current_maximum_scale + (1 - scaling_weight) * current_minimum_scale

                if free_scale:
                    target_scales = target_scales * cond_scale / target_scales.mean()

                global_multiplier = strength
                if mask_as_weight is not None:
                    global_multiplier = global_multiplier * mask_as_weight

                target_scales = target_scales * global_multiplier + torch.full_like(target_scales, cond_scale) * (1 - global_multiplier)
                conds_out[1] = make_new_uncond_at_scale(conds_out[0],conds_out[1],cond_scale,target_scales)
                return conds_out
            else:
                target_scales = maximum_scale * mask_as_weight * strength + torch.full_like(conds_out[1], cond_scale) * (1 - mask_as_weight * strength)
                conds_out[1]  = make_new_uncond_at_scale(conds_out[0],conds_out[1],cond_scale,maximum_scale)
                return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class EmptyRGBImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                              "height": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                              "g": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                              "b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                              },
                              "optional": {
                                  "grayscale_to_color": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image"
    def generate(self, width, height, batch_size=1, r=0, g=0, b=0, grayscale_to_color=None):
        if grayscale_to_color is not None:
            grayscale_to_color = grayscale_to_color.permute(0, 3, 1, 2).mean(dim=1).unsqueeze(-1)
            height = grayscale_to_color.shape[1]
            width  = grayscale_to_color.shape[2]
        r_normalized = torch.full([batch_size, height, width, 1], r / 255.0)
        g_normalized = torch.full([batch_size, height, width, 1], g / 255.0)
        b_normalized = torch.full([batch_size, height, width, 1], b / 255.0)
        rgb_image = torch.cat((r_normalized, g_normalized, b_normalized), dim=-1)
        if grayscale_to_color is not None:
            rgb_image = rgb_image * grayscale_to_color
        return (rgb_image,)

gradient_patterns = {
    "linear": lambda x, y: x,
    "sine": lambda x, y: torch.sin(x * torch.pi * y),
    "triangle": lambda x, y: 2 * torch.abs(torch.round(x % (1 / max(y, 1)) * y) - (x % (1 / max(y, 1)) * y)),
}

class GradientRGBImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 64}),
                              "height": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 64}),
                              "r1": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                              "g1": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                              "b1": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                              "r2": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                              "g2": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                              "b2": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                              "axis" : (["vertical","horizontal","circular"],),
                              "power_to": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                              "reverse_power" : ("BOOLEAN", {"default": False}),
                              },
                              "optional":{
                                  "mask": ("MASK",),
                              }
                              }
    RETURN_TYPES = ("IMAGE","MASK",)
    FUNCTION = "generate"
    CATEGORY = "image"

    def get_gradient_mask(self,width,height,horizontal):
        if horizontal:
            return torch.linspace(0, 1, width).view(1, 1, width).repeat(1, height, 1)
        return torch.linspace(0, 1, height).view(1, height, 1).repeat(1, 1, width)

    def generate(self, width, height, batch_size=1, r1=0, g1=0, b1=0, r2=255, g2=255, b2=255, pattern_value=1, power_to=1, reverse_power=False, axis="vertical", mask=None):
        gradient = self.get_gradient_mask(width, height, axis in ["horizontal","circular"])
        gradient = gradient_patterns["linear" if axis != "circular" else "sine"](gradient, pattern_value)

        if  axis == "circular":
            gradient2 = self.get_gradient_mask(width, height, False)
            gradient2 = gradient_patterns["sine"](gradient2, pattern_value)
            gradient = gradient * gradient2

        if power_to > 1:
            if reverse_power: gradient = 1 - gradient
            gradient = gradient ** power_to
            if reverse_power: gradient = 1 - gradient

        if mask is not None:
            if mask.shape != gradient.shape:
                mask = F.interpolate(mask.unsqueeze(1), size=(gradient.shape[-2], gradient.shape[-1]), mode='nearest').squeeze(1)
            gradient = gradient * mask

        gradient = gradient.squeeze(0).unsqueeze(-1)

        r_gradient = r1 / 255.0 + gradient * (r2 - r1) / 255.0
        g_gradient = g1 / 255.0 + gradient * (g2 - g1) / 255.0
        b_gradient = b1 / 255.0 + gradient * (b2 - b1) / 255.0

        r_image = r_gradient.expand(batch_size, height, width, 1)
        g_image = g_gradient.expand(batch_size, height, width, 1)
        b_image = b_gradient.expand(batch_size, height, width, 1)
        rgb_image = torch.cat((r_image, g_image, b_image), dim=-1)

        mask_gradient = gradient.expand(1, height, width, 1).squeeze(-1)

        return (rgb_image,mask_gradient,)
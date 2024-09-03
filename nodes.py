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
from comfy.taesd import taesd as taesd_class
from comfy.sample import prepare_noise
from .imported_functions import skimmed_CFG_patch_wrap
import numpy as np
import random

taesd = taesd_class.TAESD()
current_dir = os.path.dirname(os.path.realpath(__file__))
SDXL_Latent = SDXL_Latent()
sdxl_latent_rgb_factors = SDXL_Latent.latent_rgb_factors
ConditioningConcat = ConditioningConcat()
ConditioningSetTimestepRange = ConditioningSetTimestepRange()
default_attention = optimized_attention_for_device(model_management.get_torch_device())
default_device = model_management.get_torch_device()

def get_sigma_min_max(model):
    model_sampling = model.model.model_sampling
    sigma_min = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_min)).item()
    sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
    return sigma_min, sigma_max

def selfnorm(x):
    return x / x.norm()

def weighted_average(tensor1, tensor2, weight1):
    return (weight1 * tensor1 + (1 - weight1) * tensor2)

def minmaxnorm(x):
    return torch.nan_to_num((x - x.min()) / (x.max() - x.min()), nan=0.0, posinf=1.0, neginf=0.0)

def normlike(x,y):
    return x / x.norm() * y.norm()

def make_new_uncond_at_scale(cond,uncond,cond_scale,new_scale):
    new_scale_ratio = (new_scale - 1) / (cond_scale - 1)
    return cond * (1 - new_scale_ratio) + uncond * new_scale_ratio

def make_new_uncond_at_scale_co(conds_out,cond_scale,new_scale):
    new_scale_ratio = (new_scale - 1) / (cond_scale - 1)
    return conds_out[0] * (1 - new_scale_ratio) + conds_out[1] * new_scale_ratio

def get_denoised_at_scale(x_orig,cond,uncond,cond_scale):
    return x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))

@torch.no_grad()
def gaussian_kernel(size: int, sigma: float, device=default_device):
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel

def blur_tensor(tensor, kernel_size=9, sigma=1.0):
    device = tensor.device
    kernel = gaussian_kernel(kernel_size, sigma, device=device).unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    blurred_tensor = F.conv2d(tensor, kernel, padding=padding)
    return blurred_tensor.squeeze()

def roll_channel(tensor, channel_index, shift, dim):
    tensor[:, channel_index, :, :] = torch.roll(tensor[:, channel_index, :, :], shifts=shift, dims=dim)
    return tensor

def mirror_from_middle(tensor, vertical):
    dim = 2 if vertical else 3
    middle_index = tensor.size(dim) // 2
    left_part = tensor.index_select(dim, torch.arange(middle_index - 1, -1, -1, device=tensor.device))
    right_part = tensor.index_select(dim, torch.arange(middle_index, tensor.size(dim), 1, device=tensor.device))
    mirrored_tensor = torch.cat([right_part, left_part], dim=dim).to(device=tensor.device)
    return mirrored_tensor

def mirror_flip(tensor, vertical):
    dim = 2 if vertical else 3
    return torch.flip(tensor, dims=[dim])

class pre_cfg_perp_neg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "neg_scale": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,  "step": 1/10, "round": 0.01}),
                                "set_context_length" : ("BOOLEAN", {"default": False,"tooltip":"For static tensor rt engines with a set context length."}),
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
                                "empty_proportion": ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,  "step": 1/20, "round": 0.01,"tooltip":"How much of the empty prediction will be mixed with the negative.",}),
                                "progressive_scale" : ("BOOLEAN", {"default": False, "tooltip":"If turned on:\nThe proportion of empty prediction will vary along the sampling relatively to the sigma.\nThe proportion slider will set the starting value of the empty prediction.\nThe end value will be 1 - the proportion."}),
                                "set_context_length" : ("BOOLEAN", {"default": False,"tooltip":"For static tensor rt engines with a set context length."}),
                                "context_length": ("INT", {"default": 1,  "min": 1, "max": 100,  "step": 1}),
                                "end_at_sigma":   ("FLOAT", {"default": 5.42, "min": 0.0, "max": 10000.0, "step": 1/100, "round": 1/100}),
                              },
                              "optional": {
                                  "optional_text": ("STRING", {"forceInput": True,"tooltip":"If used, instead of an empty prediction, it will use this."}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, clip, empty_proportion, progressive_scale, set_context_length, context_length, end_at_sigma, optional_text=""):
        sigma_min, sigma_max = get_sigma_min_max(model)
        empty_cond, pooled = clip.encode_from_tokens(clip.tokenize(optional_text), return_pooled=True)
        nocond = [[empty_cond, {"pooled_output": pooled}]]
        if context_length > 1 and set_context_length:
            short_nocond = deepcopy(nocond)
            for x in range(context_length - 1):
                (nocond,) = ConditioningConcat.concat(nocond, short_nocond)
            nocond[0][0] = nocond[0][0][:,:77*context_length,...]
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
                                "do_on": (["both","cond","uncond"], {"default": "cond"},),
                                "scale": ("FLOAT",   {"default": 0.75, "min": -10.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                                "normalized": ("BOOLEAN", {"default": False}),
                                "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 1000000.0,  "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 1000000.0,  "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, do_on, scale, normalized, start_at_sigma, end_at_sigma):
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

            if normalized:
                n0 = conds_out[0].norm()
                if uncond:
                    n1 = conds_out[1].norm()

            prev_cond_tmp   = conds_out[0].clone()
            prev_uncond_tmp = conds_out[1].clone()

            if not first_step and sigma > end_at_sigma and sigma <= start_at_sigma:
                for b in range(len(conds_out[0])):
                    for c in range(len(conds_out[0][b])):
                        if prev_cond is not None and do_on in ['both','cond']:
                            conds_out[0][b][c] = normalize_adjust(conds_out[0][b][c], prev_cond[b][c], scale)
                        if prev_uncond is not None and uncond and do_on in ['both','uncond']:
                            conds_out[1][b][c] = normalize_adjust(conds_out[1][b][c], prev_uncond[b][c], scale)

            prev_cond = prev_cond_tmp
            if uncond:
                prev_uncond = prev_uncond_tmp

            if normalized:
                conds_out[0] = selfnorm(conds_out[0]) * n0
                if uncond:
                    conds_out[1] = selfnorm(conds_out[1]) * n1

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(sharpen_conds_pre_cfg)
        return (m, )

class condBlurSharpeningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "do_on": (["both","cond","uncond"], {"default": "both"},),
                                "operation": (["sharpen","sharpen_rescale","blur","blur_rescale"],),
                                "sharpening_scale": ("FLOAT",   {"default": 0.5, "min": -10.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                                "blur_sigma": ("FLOAT",   {"default": 0.5, "min": 0.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                                "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 1000000.0,  "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 1000000.0,  "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, do_on, operation, sharpening_scale, blur_sigma, start_at_sigma, end_at_sigma):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
        operations = {
            "sharpen": lambda x, y, z: x + (x - blur_tensor(x,sigma=z)) * y,
            "blur": lambda x, y, z: blur_tensor(x, sigma=z),
            "sharpen_rescale": lambda x, y, z: normlike(x + (x - blur_tensor(x,sigma=z)) * y, x),
            "blur_rescale": lambda x, y, z: normlike(blur_tensor(x, sigma=z), x),
        }

        @torch.no_grad()
        def sharpen_conds_pre_cfg(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])

            sigma  = args["sigma"][0].item()
            first_step = sigma > (sigma_max - 1)

            if not first_step and sigma > end_at_sigma and sigma <= start_at_sigma:
                for b in range(len(conds_out[0])):
                    for c in range(len(conds_out[0][b])):
                        if do_on in ['both','cond']:
                            conds_out[0][b][c] = operations[operation](conds_out[0][b][c],sharpening_scale,blur_sigma)
                        if uncond and do_on in ['both','uncond']:
                            conds_out[1][b][c] = operations[operation](conds_out[1][b][c],sharpening_scale,blur_sigma)

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

class PreCFGRollLatentNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "shift_per_step" : ("INT", {"default": 1,  "min": -10000, "max": 10000,  "step": 1}),
                "vertical" : ("BOOLEAN", {"default": False}),
                "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 100000.0, "step": 1/100, "round": 1/100}),
                "end_at_sigma":   ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 100000.0, "step": 1/100, "round": 1/100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, shift_per_step, vertical, start_at_sigma, end_at_sigma):
        m = model.clone()
        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            sigma = args["sigma"][0].item()
            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return conds_out
            for x in range(len(conds_out)):
                if torch.any(conds_out[x]):
                    for c in range(conds_out[x].shape[-3]):
                        conds_out[x] = roll_channel(conds_out[x], c, shift_per_step, -(1 + vertical))
            return conds_out
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m,)

class PreCFGMirrorFlipLatentNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vertical" : ("BOOLEAN", {"default": False}),
                # "do_a_flip" : ("BOOLEAN", {"default": False}),
                "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 100000.0, "step": 1/100, "round": 1/100}),
                "end_at_sigma":   ("FLOAT", {"default": 2.0,  "min": 0.0,  "max": 100000.0, "step": 1/100, "round": 1/100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, vertical, start_at_sigma, end_at_sigma, do_a_flip=True):
        m = model.clone()

        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            sigma = args["sigma"][0].item()
            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return conds_out
            for x in range(len(conds_out)):
                if torch.any(conds_out[x]):
                    if do_a_flip:
                        conds_out[x] = mirror_flip(conds_out[x], vertical)
                    else:
                        conds_out[x] = mirror_from_middle(conds_out[x], vertical)
            return conds_out
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
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
            sigma  = args["sigma"][0]

            if sigma <= end_at_sigma or sigma > start_at_sigma or all(c == 0 for c in rgb):
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

selfsquare = lambda x: x.abs().pow(2) * x.sign()

class boost_std_pre_cfg_node:
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

            for b in range(len(conds_out[0])):
                pred = 2 * conds_out[0][b] - conds_out[1][b]
                pred_std = pred.std()
                pred_std = selfsquare(pred_std)
                pred_std_cond   = conds_out[0][b].std() / pred_std
                pred_std_uncond = conds_out[1][b].std() / pred_std
                pred_std_cond   = selfsquare(pred_std_cond)
                pred_std_uncond = selfsquare(pred_std_uncond)
                conds_out[0][b] = conds_out[0][b] * pred_std_cond
                conds_out[1][b] = conds_out[1][b] * pred_std_uncond
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

@torch.no_grad()
def clamp_uncond_relative_to_sign(conds_out, same_sign, btst, value_filter, use_uncond_sign, multiplier):
    if same_sign:
        sign_mask = conds_out[0].sign() == conds_out[1].sign()
    else:
        sign_mask = conds_out[0].sign() != conds_out[1].sign()
    if   btst == "smaller_than":
        abs_mask  = conds_out[1].abs() < conds_out[0].abs()
    elif btst == "bigger_than":
        abs_mask  = conds_out[1].abs() > conds_out[0].abs()
    elif btst == "any":
        abs_mask  = sign_mask

    if value_filter == "disabled":
        value_mask = sign_mask
    elif value_filter == "only_positive_cond":
        value_mask = conds_out[0] > 0
    elif value_filter == "only_negative_cond":
        value_mask = conds_out[0] < 0
    elif value_filter == "only_positive_uncond":
        value_mask = conds_out[1] > 0
    elif value_filter == "only_negative_uncond":
        value_mask = conds_out[1] < 0

    sign_mask = sign_mask == abs_mask
    abs_sign_mask = sign_mask == value_mask

    if not same_sign and use_uncond_sign:
        result = conds_out[0][abs_sign_mask].abs() * conds_out[1][abs_sign_mask].sign()
    else:
        result = conds_out[0][abs_sign_mask]

    conds_out[1][abs_sign_mask] = conds_out[1][abs_sign_mask] + (result - conds_out[1][abs_sign_mask]) * multiplier
    return conds_out

class clamp_sign_uncond_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "same_sign" : (["disabled","bigger_than","smaller_than","any"],),
                             "opposite_sign" : (["disabled","bigger_than","smaller_than","any"],),
                             "value_filter" : (["disabled","only_positive_cond","only_negative_cond","only_positive_uncond","only_negative_uncond"],),
                             "use_uncond_sign" : ("BOOLEAN", {"default": True}),
                             "multiplier":   ("FLOAT", {"default": 1.0,  "min": -2.0, "max": 2.0,  "step": 1/2, "round": 1/100}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, same_sign, opposite_sign, value_filter, use_uncond_sign, multiplier):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            uncond = torch.any(conds_out[1])
            if not uncond or (same_sign == "disabled" and opposite_sign == "disabled"):
                return conds_out
            if same_sign != "disabled":
                conds_out = clamp_uncond_relative_to_sign(conds_out, True, same_sign, value_filter, use_uncond_sign, multiplier)
            if opposite_sign != "disabled":
                conds_out = clamp_uncond_relative_to_sign(conds_out, False, opposite_sign, value_filter, use_uncond_sign, multiplier)
            return conds_out
        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

@torch.no_grad()
def uncond_limiter(x_orig, cond, uncond, cond_scale, same_sign, opposite_sign, multiplier):
    denoised = ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))
    denoised_diff = x_orig - denoised
    cond_relation = cond_scale * cond - (cond_scale - 1) * uncond
    denoised_relation = cond_relation - denoised_diff
    uncond = uncond - denoised_relation * multiplier
    return uncond

class clamp_uncond_to_denoised_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "same_sign" : (["disabled","bigger_than","smaller_than","any"],),
                             "opposite_sign" : (["disabled","bigger_than","smaller_than","any"],),
                             "multiplier":   ("FLOAT", {"default": 1.0,  "min": -2.0, "max": 2.0,  "step": 1/2, "round": 1/100}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, same_sign, opposite_sign, multiplier):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]

            if not torch.any(conds_out[1]) or (same_sign == "disabled" and opposite_sign == "disabled"):
                return conds_out

            cond   = conds_out[0]
            uncond = conds_out[1]
            x_orig = args['input']
            cond_scale = args["cond_scale"]

            conds_out[1] = uncond_limiter(x_orig,cond,uncond,cond_scale,same_sign,opposite_sign,multiplier)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
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

class rescale_cfg_during_sigma_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "CFG_scale": ("FLOAT", {"default": 5, "min": 2.0, "max": 100.0, "step": 1/2, "round": 1/100}),
                             "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                             "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, CFG_scale, start_at_sigma, end_at_sigma):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"][0].item()
            if not torch.any(conds_out[1]) or sigma <= end_at_sigma or sigma > start_at_sigma:
                return conds_out
            if cond_scale == 1:
                cond_scale += 1e-08
            conds_out[1] = make_new_uncond_at_scale_co(conds_out,cond_scale,CFG_scale)
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

class minmax_clamp_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "cond_to_clamp": (["uncond","cond"],),
                             "enabled" : ("BOOLEAN", {"default": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, cond_to_clamp, enabled):
        clamp_neg = cond_to_clamp == "uncond"
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            uncond = torch.any(conds_out[1])
            if not uncond or not enabled:
                return conds_out
            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    conds_out[clamp_neg][b][c] = torch.clamp(conds_out[clamp_neg][b][c],min=conds_out[not clamp_neg][b][c].min(),max=conds_out[not clamp_neg][b][c].max())
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class lerp_conds_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "selection": (["both","uncond","cond"],),
                             "scale":  ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 1/10, "round": 1/100}),
                             "enabled" : ("BOOLEAN", {"default": True})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, selection, scale, enabled):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            uncond = torch.any(conds_out[1])

            if not uncond or not enabled or scale == 1:
                return conds_out

            if selection in ["both"]:
                conds_out[0], conds_out[1] = torch.lerp(conds_out[1], conds_out[0], scale), torch.lerp(conds_out[0], conds_out[1], scale)
            elif selection in ["cond"]:
                conds_out[0] = torch.lerp(conds_out[1], conds_out[0], scale)
            elif selection in ["uncond"]:
                conds_out[1] = torch.lerp(conds_out[0], conds_out[1], scale)

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

@torch.no_grad()
def random_swap(tensors, proportion=1):
    # torch.manual_seed(seed)
    num_tensors = tensors.shape[0]
    tensor_size = tensors[0].numel()

    true_count = int(tensor_size * proportion)
    mask = torch.cat((torch.ones(true_count, dtype=torch.bool, device=tensors[0].device), 
                      torch.zeros(tensor_size - true_count, dtype=torch.bool, device=tensors[0].device)))
    mask = mask[torch.randperm(tensor_size)].reshape(tensors[0].shape)
    if num_tensors == 2 and proportion < 1:
        index_tensor = torch.ones_like(tensors[0], dtype=torch.int64, device=tensors[0].device)
    else:
        index_tensor = torch.randint(1 if proportion < 1 else 0, num_tensors, tensors[0].shape, device=tensors[0].device)
    for i, t in enumerate(tensors):
        if i == 0: continue
        merge_mask = index_tensor == i & mask
        tensors[0][merge_mask] = t[merge_mask]
    return tensors[0],true_count

class gradient_scaling_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "maximum_scale": ("FLOAT", {"default": 80,  "min": 0.0, "max": 1000.0, "step": 1, "round": 1/100, "tooltip":"It is an equivalent to the CFG scale."}),
                             "minimum_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 10.0,   "step": 1/2, "round": 1/100, "tooltip":"It is an equivalent to the CFG scale."}),
                             "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 1/10, "round": 1/10}),
                             "end_at_sigma": ("FLOAT", {"default": 0.28,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                             "converging_scales" : ("BOOLEAN", {"default": True}),
                            #  "noise_add_diff" : ("BOOLEAN", {"default": True}),
                            #  "split_channels" : ("BOOLEAN", {"default": False}),
                             "invert_mask" : ("BOOLEAN", {"default": False}),
                            #  ,"black_mean","rev_mean","cond_mean"
                             "no_input" : (["rand","rev","cond","uncond","swap","r_swap","rev_swap","rev_r_swap","black","black_cond","black_uncond","black_CFG","black_CFG_diff_noise","black_CFG_x2_diff_noise","black_x2_CFG_diff_noise","black_swap","black_r_swap","all_avg","all_avg_x2","all_avg_321","all_avg_312","diff","CFG_diff","CFG_diff_noise","add_diff","rand_rev","rev_cond","rand_cond","rev_cond_sp","cond_rev_sp"],),
                            #  "start_at_sigma": ("FLOAT", {"default": 15,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                            #  "end_at_sigma": ("FLOAT", {"default": 0.28,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                            "free_scale" : (["disabled","sharp","sharp_clamp","full","full_blur","full_sharp","full_clamp","full_blur_clamp","full_sharp_clamp"],),
                            # "free_scale" : (["disabled","full","per_layer","full_and_blur","per_layer_and_blur","full_clamp","per_layer_clamp","full_and_blur_clamp","per_layer_and_blur_clamp"],),
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

    def get_black_latent(self,):
        latent_path = os.path.join(current_dir,"latents","sdxl_black.pt")
        latent_space = torch.load(latent_path).to(device=default_device)
        return latent_space

    def patch(self, model, maximum_scale, minimum_scale, strength, end_at_sigma, start_at_sigma=99999, invert_mask=False, no_input="black", noise_add_diff=True, converging_scales=False, split_channels=False, free_scale="disabled", input_mask=None, input_latent=None):
        black_avg = [-21.758529663085938, 3.8702831268310547, 2.311274766921997, 2.559422016143799]
        black_latent = self.get_black_latent()
        # if input_mask is None and input_latent is None:
        #     return (model,)
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

        @torch.no_grad()
        def snc(x): return x / x.norm()
        trl = lambda x: torch.randn_like(x,device=x.device)

        # p y t h o n i c  a s  f u c k  >.<
        no_input_operations = {
            "rand": lambda x, y, o, z, s, c: snc(trl(x)) * x.norm(),
            "rev": lambda x, y, o, z, s, c: x * -1,
            "cond": lambda x, y, o, z, s, c: snc(y) * x.norm(),
            "uncond": lambda x, y, o, z, s, c: snc(o) * x.norm() * -1,
            "swap": lambda x, y, o, z, s, c: no_input_operations["cond"](x, y, o, z, s, c) if s > 0.37 else no_input_operations["uncond"](x, y, o, z, s, c),
            "r_swap": lambda x, y, o, z, s, c: no_input_operations["cond"](x, y, o, z, s, c) if s <= 0.37 else no_input_operations["uncond"](x, y, o, z, s, c),
            "rev_swap": lambda x, y, o, z, s, c: snc(no_input_operations["rev"](x, y, o, z, s, c)+no_input_operations["swap"](x, y, o, z, s, c)) * x.norm(),
            "rev_r_swap": lambda x, y, o, z, s, c: snc(no_input_operations["rev"](x, y, o, z, s, c)+no_input_operations["r_swap"](x, y, o, z, s, c)) * x.norm(),
            # "black": lambda x, y, o, z, s, c: snc(torch.tensor(black_avg).view(1, len(black_avg), 1, 1).expand(1, len(black_avg), x.shape[2], x.shape[3]).to(device=x.device)) * x.norm(),
            "black": lambda x, y, o, z, s, c: snc(black_latent.clone()) * x.norm(),
            "black_cond": lambda x, y, o, z, s, c: (no_input_operations["cond"](x, y, o, z, s, c) + no_input_operations["black"](x, y, o, z, s, c)) / 2,
            "black_uncond": lambda x, y, o, z, s, c: (no_input_operations["uncond"](x, y, o, z, s, c) + no_input_operations["black"](x, y, o, z, s, c)) / 2,
            "black_CFG": lambda x, y, o, z, s, c: (snc(y * c - o * (c - 1)) * x.norm() + no_input_operations["black"](x, y, o, z, s, c)) / 2,
            "black_denoised": lambda x, y, o, z, s, c: (snc(x - ( (x - o) + c * ( (x - y) - (x - o) ))) * x.norm() + no_input_operations["black"](x, y, o, z, s, c)) / 2,
            "black_denoised_x2": lambda x, y, o, z, s, c: (snc(x - ( (x - o) + c * ( (x - y) - (x - o) ))) * x.norm() * 0.5 + no_input_operations["black"](x, y, o, z, s, c) * 1.5) / 2,
            "black_CFG_diff_noise": lambda x, y, o, z, s, c: snc(no_input_operations["black_CFG"](x, y, o, z, s, c)-x) * x.norm(),
            "black_CFG_x2_diff_noise": lambda x, y, o, z, s, c: snc(2 * no_input_operations["black_CFG"](x, y, o, z, s, c) - x) * x.norm(),
            "black_x2_CFG_diff_noise": lambda x, y, o, z, s, c: snc(no_input_operations["black"](x, y, o, z, s, c) + no_input_operations["black_CFG"](x, y, o, z, s, c) - x) * x.norm(),
            "black_swap": lambda x, y, o, z, s, c: (no_input_operations["black"](x, y, o, z, s, c) + no_input_operations["swap"](x, y, o, z, s, c)) / 2,
            "black_r_swap": lambda x, y, o, z, s, c: (no_input_operations["black"](x, y, o, z, s, c) + no_input_operations["r_swap"](x, y, o, z, s, c)) / 2,
            "all_avg": lambda x, y, o, z, s, c: snc(y-x-o) * x.norm(),
            "all_avg_x2": lambda x, y, o, z, s, c: snc(2*y-x-o) * x.norm(),
            "all_avg_321": lambda x, y, o, z, s, c: snc(3*y-2*x-o) * x.norm(),
            "all_avg_312": lambda x, y, o, z, s, c: snc(3*y-2*o-x) * x.norm(),
            "diff": lambda x, y, o, z, s, c: snc(y - o) * x.norm(),
            "CFG_diff": lambda x, y, o, z, s, c: snc(y * c - o * (c - 1)) * x.norm(),
            "CFG_diff_noise": lambda x, y, o, z, s, c: (snc(y * c - o * (c - 1)) + snc(-x)) / 2 * x.norm(),
            "add_diff": lambda x, y, o, z, s, c: snc(y + (y - o) / 2) * x.norm(),
            "rand_rev": lambda x, y, o, z, s, c: snc(trl(x)) * x.norm(),
            "rev_cond": lambda x, y, o, z, s, c: (snc(x) * -1 + snc(y) * 0.5) * x.norm() / 1.5,
            "rand_cond": lambda x, y, o, z, s, c: (snc(x) * -1 + snc(trl(x)) * 0.5) * x.norm() / 1.5,
            "rev_cond_sp": lambda x, y, o, z, s, c: no_input_operations["rev"](x, y, o, z, s, c) * z + (1 - z) * no_input_operations["cond"](x, y, o, z, s, c),
            "cond_rev_sp": lambda x, y, o, z, s, c: no_input_operations["rev"](x, y, o, z, s, c) * (1 - z) + z * no_input_operations["cond"](x, y, o, z, s, c),
        }

        @torch.no_grad()
        def pre_cfg_patch(args):
            nonlocal mask_as_weight, latent_as_guidance, black_latent
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args['input']
            sigma  = args["sigma"][0]
            sp = min(1,max(0,sigma_to_percent(model_sampling, sigma - sigma_min * 3) + 1 / 100)) ** 2
            sp2 = (sigma - sigma_min) / (sigma_max - sigma_min)

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

            if "black" in no_input and random_guidance:
                if black_latent.shape[-2:] != conds_out[1].shape[-2:]:
                    black_latent = F.interpolate(black_latent, size=(conds_out[1].shape[-2], conds_out[1].shape[-1]), mode='bilinear', align_corners=False)

            if random_guidance:
                latent_as_guidance = no_input_operations[no_input](x_orig.clone(),conds_out[0].clone(),conds_out[1].clone(),sp,sp2,cond_scale)

            if latent_as_guidance is not None:
                if latent_as_guidance.shape[-2:] != conds_out[1].shape[-2:]:
                    latent_as_guidance = F.interpolate(latent_as_guidance, size=(conds_out[1].shape[-2], conds_out[1].shape[-1]), mode='bilinear', align_corners=False)

                scaling_weight = scaling_function(x_orig,conds_out[0],conds_out[1],latent_as_guidance.clone(),current_minimum_scale,current_maximum_scale,noise_add_diff)

                target_scales = scaling_weight * current_maximum_scale + (1 - scaling_weight) * current_minimum_scale

                if "blur" in free_scale:
                    for b in range(target_scales.shape[0]):
                        for c in range(target_scales.shape[1]):
                            target_scales[b][c] = blur_tensor(target_scales[b][c], kernel_size=9, sigma=1.0)
                elif "sharp" in free_scale:
                    for b in range(target_scales.shape[0]):
                        for c in range(target_scales.shape[1]):
                            target_scales[b][c] = target_scales[b][c] + (target_scales[b][c] - blur_tensor(target_scales[b][c], kernel_size=9, sigma=0.5)) * 0.5

                if "full" in free_scale:
                    target_scales = target_scales * cond_scale / target_scales.mean()
                elif "per_layer" in free_scale:
                    for b in range(target_scales.shape[0]):
                        for c in range(target_scales.shape[1]):
                            target_scales[b][c] = target_scales[b][c] * cond_scale / target_scales[b][c].mean()

                if "clamp" in free_scale:
                    target_scales = torch.clamp(target_scales,min=current_minimum_scale,max=current_maximum_scale)

                global_multiplier = strength
                if input_mask is not None:
                    global_multiplier = global_multiplier * mask_as_weight

                target_scales = target_scales * global_multiplier + torch.full_like(target_scales, cond_scale) * (1 - global_multiplier)
                conds_out[1] = make_new_uncond_at_scale(conds_out[0],conds_out[1],cond_scale,target_scales)
                return conds_out
            else:
                target_scales = maximum_scale * mask_as_weight * strength + torch.full_like(conds_out[1], cond_scale) * (1 - mask_as_weight * strength)
                conds_out[1]  = make_new_uncond_at_scale(conds_out[0],conds_out[1],cond_scale,target_scales)
                return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

dArK_ScRiPtUrE = lambda s: ''.join(random.choice([c.upper(), c.lower()]) for c in s)
dArK_mEtHoDs = {"ThE dArKnEsS":["black",1000,"will come and remaiiiin",True,"full_blur",[]],
                "ThE CoNtRaCt":["black_denoised",1000,"has been siiiiiiigned.",True,"full_blur",[]],
                "ThE ExChAnGe":["black_swap",1000,"will be done with his souuuuuuul",True,"full_blur",[]],
                "ThE rEdEmPtiOn":["swap",16,"is never pooossiiiibleeeee.",False,"disabled",[]],
                "ThE ShAdOw":["black",12,"of his mom covers the eaaaarth",False,"disabled",[]],
                "ThE AbYsS":["black",32,"is deeeeeeeeeeeeeeeeeep",False,"disabled",[]],
                "ThE WhIsPeR oF tHe DrOwNeD":["swap",12,"sooooundeeeeed liiiiiiiiiiike 'blub'.",False,"disabled",[["ThE ShAdOw", False]]],
                }

dArK_nAmEs = [n for n in dArK_mEtHoDs]
class dark_guidance_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "RiTuaL": (dArK_nAmEs, {"default": dArK_nAmEs[0]}),
                             "AsK_fOr_ForGiVeNeSs" : ("BOOLEAN", {"default": False}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Pre CFG"
    def patch(self, model, RiTuaL, AsK_fOr_ForGiVeNeSs, as_extra=False):
        if not as_extra:
            decided = dArK_ScRiPtUrE(f"{RiTuaL} {dArK_mEtHoDs[RiTuaL][2]}")
            print(f" \033[91m\033[4m{decided}\033[0m")
        simi,sima = get_sigma_min_max(model)
        # short_end = (sima-simi)*0.0172+simi
        guided_guidance_scaling = gradient_scaling_pre_cfg_node()
        m, = guided_guidance_scaling.patch(model=model, end_at_sigma=0,maximum_scale=dArK_mEtHoDs[RiTuaL][1],minimum_scale=4.75,
                                          strength=1,converging_scales=True,no_input=dArK_mEtHoDs[RiTuaL][0],free_scale=dArK_mEtHoDs[RiTuaL][4])
        if AsK_fOr_ForGiVeNeSs:
            m, = skimmed_CFG_patch_wrap(m,end_proportion=0.35)
        if dArK_mEtHoDs[RiTuaL][3]:
            cds = condDiffSharpeningNode()
            m, = cds.patch(model=m,do_on="both",start_at_sigma=99999,end_at_sigma=0,scale=0.3,normalized=True)
            cdsb = condBlurSharpeningNode()
            m, = cdsb.patch(model=m,do_on="both",start_at_sigma=99999,end_at_sigma=(sima-simi)*0.37+simi,
                            sharpening_scale=0.5,blur_sigma=0.5,operation="sharpen_rescale")

        for rit in dArK_mEtHoDs[RiTuaL][5]:
            dcg1 = dark_guidance_pre_cfg_node()
            m, = dcg1.patch(m,RiTuaL=rit[0],AsK_fOr_ForGiVeNeSs=rit[1],as_extra=True)
        return (m, )

def adjust_vibrance(images: torch.Tensor, vibrance_factor: float) -> torch.Tensor:
    grayscale_images = images.mean(dim=-1, keepdim=True)
    vibrance_adjustment = (images - grayscale_images) * vibrance_factor
    low_saturation_mask = (images - grayscale_images).abs() < 0.5
    adjusted_images = images + vibrance_adjustment * low_saturation_mask.float()
    return torch.clamp(adjusted_images, 0, 1)

def adjust_saturation(images: torch.Tensor, vibrance_factor: float) -> torch.Tensor:
    grayscale_images = images.mean(dim=-1, keepdim=True)
    adjusted_images = grayscale_images + vibrance_factor * (images - grayscale_images)
    return torch.clamp(adjusted_images, 0, 1)

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

# red = [-19.2851, -19.4045, 11.3942, -11.8191]
# green = [-3.2465, 14.3003, 27.0502, 9.1685]
# blue = [0.5476, 16.1999, -17.2144, 4.3121]

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

def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

class gradientNoisyLatentMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "denoising_start": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "denoising_end": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "inject_noise" : (["disabled","gradient","full_for_first_batch"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sigmas": ("SIGMAS",),
            }
            }
    RETURN_TYPES = ("MODEL","LATENT",)
    RETURN_NAMES = ("model","noisy_latents_for_sampler",)
    FUNCTION = "generate"
    CATEGORY = "model_patches"
    
    def generate(self, model, latent, denoising_start, denoising_end, inject_noise, seed, sigmas):
        sigma_min, sigma_max = get_sigma_min_max(model)
        input_sigma_min = (sigmas[sigmas>0]).min()
        samples = latent["samples"].clone()
        if samples.shape[0] < denoising_end:
            additional_latents = latent["samples"][-1:].repeat(denoising_end - samples.shape[0], 1, 1, 1)
            samples = torch.cat((samples, additional_latents), dim=0)

        if inject_noise != "disabled":
            noisy_latents = prepare_noise(torch.zeros_like(samples, device=samples.device), seed)
            # .unsqueeze(0)
            # .repeat(samples.shape[0], 1, 1, 1)

        batched_sigmas = []
        if inject_noise == "gradient":
            cropped_sigmas = sigmas[:-1]
            for x in range(samples.shape[0]):
                if x < denoising_start:
                    batched_sigmas.append(torch.zeros_like(sigmas).to(device=default_device))
                    continue
                first_step = (x - denoising_start + 1) / (denoising_end - denoising_start)
                batch_sigmas = loglinear_interp(cropped_sigmas.tolist(),floor(cropped_sigmas.shape[0] / first_step))[::-1][:cropped_sigmas.shape[0]][::-1]
                batch_sigmas = torch.tensor(batch_sigmas)
                batch_sigmas = torch.cat([batch_sigmas, torch.tensor([0.])])
                sigmasf = float((batch_sigmas[0]-batch_sigmas[-1])/model.model.latent_format.scale_factor)
                samples[x] = samples[x] + noisy_latents[x] * sigmasf
                batched_sigmas.append(batch_sigmas.to(device=default_device))
            # for cb in batched_sigmas:
            #     print(cb,cb.shape)
        elif inject_noise == "full_for_first_batch":
            sigmasf = float((sigmas[0]-sigmas[-1])/model.model.latent_format.scale_factor)
            samples = (noisy_latents * sigmasf)

        current_step = 0
        @torch.no_grad()
        def cfg_patch(args):
            nonlocal current_step
            cond     = args["cond"]
            uncond   = args["uncond"]
            scale    = args["cond_scale"]
            sigma    = args["sigma"][0].item()

            denoised = uncond + scale * (cond - uncond)

            if inject_noise != "gradient" or sigma == 0:
                return denoised

            if sigma > (sigma_max - 1):
                current_step = 0

            for b in range(len(cond)):
                if b == len(cond) - 1:
                    continue
                print("-"*40)
                print(sigma,batched_sigmas[-1][current_step].item(),batched_sigmas[b][current_step].item())
                batch_ratio = batched_sigmas[b][current_step] / batched_sigmas[-1][current_step]
                denoised[b] = denoised[b] * batch_ratio
                # denoised[b] = (denoised[b] - sigma_min) * batch_ratio + sigma_min
                # denoised[b] = (denoised[b] - input_sigma_min) * batch_ratio + input_sigma_min
            current_step += 1
            return denoised

        m = model.clone()
        m.set_model_sampler_cfg_function(cfg_patch)
        return (m,{"samples":samples},)
    
gradient_patterns = {
    "linear": lambda x, y: x,
    "sine": lambda x, y: torch.sin(x * torch.pi * y),
    "triangle": lambda x, y: 2 * torch.abs(torch.round(x % (1 / max(y, 1)) * y) - (x % (1 / max(y, 1)) * y)),
}

class load_latent_for_guidance:
    @classmethod
    def INPUT_TYPES(s):
        latents_folder = os.path.join(current_dir,"latents")
        latents_names  = os.listdir(latents_folder)
        return {"required": {
                             "latent_name": (latents_names,)
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "exec"

    CATEGORY = "latent"

    def exec(self, latent_name):
        latent_path = os.path.join(current_dir,"latents",latent_name)
        latent_space = torch.load(latent_path).cpu()
        latent = {"samples": latent_space.unsqueeze(0)}
        return (latent, )

# >>> x=torch.rand(1,50,50,3)
# >>> x.permute(0, 3, 1, 2).shape
# torch.Size([1, 3, 50, 50])
# >>> x.permute(0, 3, 1, 2).permute(0, 2, 3, 1).shape
# torch.Size([1, 50, 50, 3])

# rgb black white
sdxl_latent_full_colors = torch.tensor([
    [-19.467044830322266, -19.590354919433594, 10.634221076965332, -12.227653503417969],
    [-3.2475805282592773, 14.238205909729004, 26.689006805419922, 9.03557300567627],
    [0.5756417512893677, 16.23186492919922, -16.991405487060547, 4.4524736404418945],
    [-21.758529663085938, 3.8702831268310547, 2.311274766921997, 2.559422016143799],
    [18.211639404296875, 1.7760906219482422, 9.4437255859375, -7.949795722961426]])

# def make_latent_color(lat,r,g,b):
#     col = sdxl_latent_full_colors.clone()
#     w   = (r+g+b) / 3
#     bw  = col[3] * (1 - w) + col[4] * w
#     lat = lat + col[0]*r + col[1] * g + col[2] * b + bw
#     for b in range(lat.shape[0]):
#         for c in range(lat.shape[1]):
#             bw  = col[3][c] * (1 - w) + col[4][c] * w
#             lat[b][c] = lat[b][c] + col[0][c] * r + col[1][c] * g + col[2][c] * b + bw
#     return lat

def make_latent_color(lat, r, g, b):
    col = sdxl_latent_full_colors.clone()
    w = (r + g + b) / 3
    # bw = col[3].view(1, 4, 1, 1) * (1 - w) + col[4].view(1, 4, 1, 1) * w
    # lat = lat + col[0].view(1, 4, 1, 1) * r + col[1].view(1, 4, 1, 1) * g + col[2].view(1, 4, 1, 1) * b + bw
    black = col[3].view(1, 4, 1, 1) * (1 - w)
    white = col[4].view(1, 4, 1, 1) * w
    red   = col[0].view(1, 4, 1, 1) * r
    green = col[1].view(1, 4, 1, 1) * g
    blue  = col[2].view(1, 4, 1, 1) * b
    lat = lat + (red/1.45 + green/1.45 + blue) / 3
    return lat

class colors_test_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "width": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 64}),
            "height": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 64}),
            "r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            "g": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            "b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "exec"
    CATEGORY = "latent"
    def exec(self, batch_size, width, height, r,g,b):
        new_latent = torch.zeros([batch_size,4,height//8,width//8])
        new_latent = make_latent_color(new_latent,r,g,b)
        # latent_path = os.path.join(current_dir,"latents",latent_name)
        # latent_space = torch.load(latent_path).cpu()
        # latent = {"samples": latent_space.unsqueeze(0)}
        return ({"samples": new_latent}, )

class latent_recombine_channels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "channel1": ("LATENT",),
                             "channel2": ("LATENT",),
                             "channel3": ("LATENT",),
                             "channel4": ("LATENT",),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "exec"
    CATEGORY = "latent"

    def exec(self, channel1, channel2, channel3, channel4):
        min_batch = min(channel1["samples"].shape[0],channel2["samples"].shape[0],
                        channel3["samples"].shape[0],channel4["samples"].shape[0])
        latents1 = channel1["samples"].clone()
        latents1[0:min_batch,1,:,:] = channel2["samples"][0:min_batch,1,:,:]
        latents1[0:min_batch,2,:,:] = channel3["samples"][0:min_batch,2,:,:]
        latents1[0:min_batch,3,:,:] = channel4["samples"][0:min_batch,3,:,:]
        output_samples = {"samples":latents1}
        return (output_samples, )
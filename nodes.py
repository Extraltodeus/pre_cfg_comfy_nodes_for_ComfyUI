import torch
from comfy.samplers import calc_cond_batch, encode_model_conds
from comfy.sampler_helpers import convert_cond

class pre_cfg_perp_neg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "neg_scale": ("FLOAT",   {"default": 1.0,  "min": 0.0, "max": 10.0,  "step": 1/10, "round": 0.01}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, clip, neg_scale):
        empty_cond, pooled = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        nocond = convert_cond([[empty_cond, {"pooled_output": pooled}]])
        
        @torch.no_grad()
        def pre_cfg_perp_neg_function(args):
            conds_out = args["conds_out"]
            noise_pred_pos = conds_out[0]

            if torch.any(conds_out[1]):
                noise_pred_neg = conds_out[1]
            else:
                return conds_out

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

selfnorm = lambda x: x / x.norm()
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
                                "scale":     ("FLOAT",   {"default": 0.75, "min": -10.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, scale):
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

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    if not first_step and sigma > 1:
                        if prev_cond is not None:
                            conds_out[0][b][c]   = normalize_adjust(conds_out[0][b][c], prev_cond[b][c], scale)
                        if prev_uncond is not None and uncond:
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
                                "exponent": ("FLOAT",   {"default": 0.8, "min": 0.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, exponent):
        @torch.no_grad()
        def exponentiate_conds_pre_cfg(args):
            if args["sigma"][0] <= 1: return args["conds_out"]

            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])

            for b in range(len(conds_out[0])):
                conds_out[0][b] = normalized_pow(conds_out[0][b], exponent)
                if uncond:
                    conds_out[1][b] = normalized_pow(conds_out[1][b], exponent)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(exponentiate_conds_pre_cfg)
        return (m, )

@torch.no_grad()
def topk_average(latent, top_k=0.25):
    max_values = torch.topk(latent, k=int(len(latent)*top_k), largest=True).values
    min_values = torch.topk(latent, k=int(len(latent)*top_k), largest=False).values
    max_val = torch.mean(max_values).item()
    min_val = torch.mean(torch.abs(min_values)).item()
    value_range = (max_val + min_val) / 2
    return value_range

class automatic_pre_cfg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "mode": (["automatic_cfg","strict_scaling"],),
                                "support_empty_uncond" : ("BOOLEAN", {"default": False}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, mode, support_empty_uncond):
        @torch.no_grad()
        def automatic_pre_cfg(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            cond_scale = args["cond_scale"]
            
            if not uncond:
                if support_empty_uncond:
                    conds_out[0] /= cond_scale
                return conds_out

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    mes = topk_average(8 * conds_out[0][b][c] - 7 * conds_out[1][b][c])

                    if mode == "automatic_cfg":
                        new_scale = 0.8 / max(mes,0.01)
                    else:
                        new_scale = cond_scale / max(mes * 8, 0.01)

                    conds_out[0][b][c] *= new_scale
                    conds_out[1][b][c] *= new_scale

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(automatic_pre_cfg)
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
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, scale):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
        prev_conds   = []
        prev_unconds = []

        @torch.no_grad()
        def sharpen_conds_pre_cfg(args):
            nonlocal prev_conds, prev_unconds
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            
            sigma  = args["sigma"][0].item()

            if sigma <= 1:
                return conds_out

            first_step = sigma > (sigma_max - 1)
            if first_step:
                prev_conds   = []
                prev_unconds = []

            prev_conds.append(conds_out[0] / conds_out[0].norm())
            if uncond:
                prev_unconds.append(conds_out[1] / conds_out[1].norm())

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
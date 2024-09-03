import torch

@torch.no_grad()
def get_skimming_mask(x_orig, cond, uncond, cond_scale, return_denoised=False, disable_flipping_filter=False, release_inner_scaling=False):
    denoised = x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = (denoised.sign() == (denoised - x_orig).sign())
        outer_influence = matching_pred_signs & matching_diff_after & deviation_influence
    if return_denoised:
        return outer_influence, denoised
    else:
        return outer_influence
    
@torch.no_grad()
def skimmed_CFG(x_orig, cond, uncond, cond_scale, skimming_scale, disable_flipping_filter=False):
    outer_influence, denoised = get_skimming_mask(x_orig, cond, uncond, cond_scale, return_denoised=True, disable_flipping_filter=disable_flipping_filter)
    low_cfg_denoised_outer = x_orig - ((x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond)))
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (low_cfg_denoised_outer_difference[outer_influence] / cond_scale)
    return cond

def skimmed_CFG_patch_wrap(model,Skimming_CFG=-1,end_proportion=1,full_skim_negative=True,disable_flipping_filter=False):
    @torch.no_grad()
    def skimmed_CFG_patch(args):
        conds_out  = args["conds_out"]
        cond_scale = args["cond_scale"]
        x_orig     = args['input']
        if not torch.any(conds_out[1]):
            return conds_out
        if end_proportion != 1:
            c0,c1=conds_out[0].clone(),conds_out[1].clone()
        practical_scale = cond_scale if Skimming_CFG < 0 else Skimming_CFG
        conds_out[1] = skimmed_CFG(x_orig, conds_out[1], conds_out[0], cond_scale, practical_scale if not full_skim_negative else 0, disable_flipping_filter)
        conds_out[0] = skimmed_CFG(x_orig, conds_out[0], conds_out[1], cond_scale, practical_scale, disable_flipping_filter)
        if end_proportion != 1:
            conds_out[0] = conds_out[0] * end_proportion + c0 * (1 - end_proportion)
            conds_out[1] = conds_out[1] * end_proportion + c1 * (1 - end_proportion)
        return conds_out
    m = model.clone()
    m.set_model_sampler_pre_cfg_function(skimmed_CFG_patch)
    return m,

# def skimmed_CFG_patch_wrap(model,Skimming_CFG=-1,end_proportion=1,full_skim_negative=False,disable_flipping_filter=False):
#     @torch.no_grad()
#     def skimmed_CFG_patch(args):
#         conds_out  = args["conds_out"]
#         cond_scale = args["cond_scale"]
#         x_orig     = args['input']
#         if not torch.any(conds_out[1]):
#             return conds_out
#         if end_proportion != 1:
#             c0,c1=conds_out[0].clone(),conds_out[1].clone()
#         practical_scale = cond_scale if Skimming_CFG < 0 else Skimming_CFG
#         conds_out[1] = skimmed_CFG(x_orig, conds_out[1], conds_out[0], cond_scale, practical_scale if not full_skim_negative else 0, disable_flipping_filter)
#         conds_out[0] = skimmed_CFG(x_orig, conds_out[0], conds_out[1], cond_scale, practical_scale, disable_flipping_filter)
#         if end_proportion != 1:
#             conds_out[0] = conds_out[0] * end_proportion + c0 * (1 - end_proportion)
#             conds_out[1] = conds_out[1] * end_proportion + c1 * (1 - end_proportion)
#         return conds_out
#     m = model.clone()
#     m.set_model_sampler_pre_cfg_function(skimmed_CFG_patch)
#     return m,
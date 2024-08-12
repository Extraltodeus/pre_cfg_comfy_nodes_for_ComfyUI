from .nodes import *

NODE_CLASS_MAPPINGS = {}

NODE_CLASS_MAPPINGS_ADD = {
    "Pre CFG automatic scale": automatic_pre_cfg,
    "Pre CFG uncond zero": uncondZeroPreCFGNode,
    "Pre CFG perp-neg": pre_cfg_perp_neg,
    # "Pre CFG re-negative": pre_cfg_re_negative,
    
    "Pre CFG PAG": perturbed_attention_guidance_pre_cfg_node,
    "Pre CFG zero attention": zero_attention_pre_cfg_node,
    "Pre CFG channel multiplier": channel_multiplier_node,
    "Pre CFG multiplier": multiply_cond_pre_cfg_node,

    "Pre CFG norm neg to pos": norm_uncond_to_cond_pre_cfg_node,
    "Pre CFG subtract mean": PreCFGsubtractMeanNode,
    "Pre CFG variable scaling": variable_scale_pre_cfg_node,
    "Pre CFG gradient scaling": gradient_scaling_pre_cfg_node,

    "Pre CFG flip flop": flip_flip_conds_pre_cfg_node,
    "Pre CFG replace negative channel": replace_uncond_channel_pre_cfg_node,
    "Pre CFG merge negative channel": merge_uncond_channel_pre_cfg_node,    

    "Pre CFG sharpening": condDiffSharpeningNode,
    "Pre CFG exponentiation": condExpNode,

    "Conditioning set timestep from sigma": ConditioningSetTimestepRangeFromSigma,
    "Support empty uncond": support_empty_uncond_pre_cfg_node,
    "Shape attention": ShapeAttentionNode,
    "Excellent attention": ExlAttentionNode,
    "Post CFG subtract mean": PostCFGsubtractMeanNode,
    "Individual channel selector": individual_channel_selection_node,
    "Subtract noise mean": latent_noise_subtract_mean_node,
    "Empty RGB image": EmptyRGBImage,
    "Gradient RGB image": GradientRGBImage,
}

NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_ADD)

for c in [4,8,16,32,64,128]:
    NODE_CLASS_MAPPINGS[f"Channel selector for {c} channels"] = type("channel_selection_node", (channel_selection_node,), { "CHANNELS_AMOUNT": c})

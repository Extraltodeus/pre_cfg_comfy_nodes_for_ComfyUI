from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Pre CFG automatic scale": automatic_pre_cfg,
    "Pre CFG uncond zero": uncondZeroPreCFGNode,
    "Pre CFG perp-neg": pre_cfg_perp_neg,
    "Pre CFG PAG": perturbed_attention_guidance_pre_cfg_node,
    "Pre CFG zero attention": zero_attention_pre_cfg_node,
    # "Pre CFG color control": latent_color_control_pre_cfg_node,
    "Pre CFG channel multiplier": channel_multiplier_node,
    "Pre CFG subtract mean": PreCFGsubtractMeanNode,
    "Pre CFG sharpening": condDiffSharpeningNode,
    "Pre CFG consensus sharpening": condConsensusSharpeningNode,
    "Pre CFG exponentiation": condExpNode,
    "Pre CFG variable scaling": variable_scale_pre_cfg_node,
    
    "Conditioning set timestep from sigma": ConditioningSetTimestepRangeFromSigma,
    "Support empty uncond": support_empty_uncond_pre_cfg_node,
    "Shape attention": ShapeAttentionNode,
    "Post CFG subtract mean": PostCFGsubtractMeanNode,
    "Post CFG make a dot": PostCFGDotNode,
    "Individual channel selector": individual_channel_selection_node,
    
}

for c in [4,8,16,32,64,128]:
    NODE_CLASS_MAPPINGS[f"Channel selector for {c} channels"] = type("channel_selection_node", (channel_selection_node,), { "CHANNELS_AMOUNT": c})
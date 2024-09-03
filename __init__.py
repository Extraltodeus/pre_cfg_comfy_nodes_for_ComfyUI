from .nodes import *

NODE_CLASS_MAPPINGS = {}

# try:
#     from .skimmed_CFG import cond_skimming_pre_cfg_node
#     NODE_CLASS_MAPPINGS["Skimmed CFG"] = cond_skimming_pre_cfg_node
# except:
#     pass

NODE_CLASS_MAPPINGS_ADD = {
    "Pre CFG automatic scale": automatic_pre_cfg,
    "Pre CFG uncond zero": uncondZeroPreCFGNode,
    "Pre CFG perp-neg": pre_cfg_perp_neg,
    "Pre CFG re-negative": pre_cfg_re_negative,
    
    "Pre CFG PAG": perturbed_attention_guidance_pre_cfg_node,
    "Pre CFG zero attention": zero_attention_pre_cfg_node,
    # "Pre CFG color control": latent_color_control_pre_cfg_node,
    "Pre CFG channel multiplier": channel_multiplier_node,
    "Pre CFG multiplier": multiply_cond_pre_cfg_node,
    "Pre CFG roll latent": PreCFGRollLatentNode,
    "Pre CFG mirror flip": PreCFGMirrorFlipLatentNode,

    "Pre CFG clamp negative": clamp_sign_uncond_pre_cfg_node,
    "Pre CFG clamp negative to denoised relation": clamp_uncond_to_denoised_pre_cfg_node,

    "Pre CFG clamp min max": minmax_clamp_pre_cfg_node,
    "Pre CFG lerp": lerp_conds_pre_cfg_node,
    
    "Pre CFG norm neg to pos": norm_uncond_to_cond_pre_cfg_node,
    "Pre CFG subtract mean": PreCFGsubtractMeanNode,
    "Pre CFG variable scaling": variable_scale_pre_cfg_node,
    "Pre CFG gradient scaling": gradient_scaling_pre_cfg_node,
    
    "Pre CFG flip flop": flip_flip_conds_pre_cfg_node,
    "Pre CFG replace negative channel": replace_uncond_channel_pre_cfg_node,
    "Pre CFG merge negative channel": merge_uncond_channel_pre_cfg_node,
    "Pre CFG timed CFG rescale": rescale_cfg_during_sigma_pre_cfg_node,

    "Pre CFG sharpening": condDiffSharpeningNode,
    "Pre CFG sharpen/blur": condBlurSharpeningNode,
    
    "Pre CFG exponentiation": condExpNode,
    "Pre CFG cond boost": boost_std_pre_cfg_node,
    "tHe dArK GuiDaNcE": dark_guidance_pre_cfg_node,

    "Conditioning set timestep from sigma": ConditioningSetTimestepRangeFromSigma,
    "Support empty uncond": support_empty_uncond_pre_cfg_node,
    "Shape attention": ShapeAttentionNode,
    "Excellent attention": ExlAttentionNode,
    "Post CFG subtract mean": PostCFGsubtractMeanNode,
    # "Post CFG make a dot": PostCFGDotNode,
    "Individual channel selector": individual_channel_selection_node,
    "Subtract noise mean": latent_noise_subtract_mean_node,
    "Empty RGB image": EmptyRGBImage,
    "Gradient RGB image": GradientRGBImage,
    "colors test node": colors_test_node,
    "gradient batch mask": gradientNoisyLatentMaskBatch,
    "Load latent from path": load_latent_for_guidance,
    "Latent recombine by channels": latent_recombine_channels,
}

NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_ADD)

for c in [4,8,16,32,64,128]:
    NODE_CLASS_MAPPINGS[f"Channel selector for {c} channels"] = type("channel_selection_node", (channel_selection_node,), { "CHANNELS_AMOUNT": c})

try:
    from .tester_nodes import *
    NODE_CLASS_MAPPINGS["K-K-K-K-KOMBO BREAKER"] = combo_breaker
    NODE_CLASS_MAPPINGS["K-K-K-K-KOMBO BREAKER X2"] = combo_breaker_x2
    
    NODE_CLASS_MAPPINGS["K-K-K-K-KOMBO BREAKER 6 bool"] = combo_breaker_6_bool
    NODE_CLASS_MAPPINGS["K-K-K-K-KOMBO BREAKER 4 bool"] = combo_breaker_4_bool
    
    NODE_CLASS_MAPPINGS["CFG_TEST"] = cfg_test_node

except:
    pass


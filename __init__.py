from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Pre CFG perp-neg": pre_cfg_perp_neg,
    "Pre CFG sharpening": condDiffSharpeningNode,
    "Pre CFG consensus sharpening": condConsensusSharpeningNode,
    "Pre CFG exponentiation": condExpNode,
    "Pre CFG automatic scale": automatic_pre_cfg,
}

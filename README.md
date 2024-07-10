# Pre CFG nodes

A set of nodes to prepare the noise predictions before the CFG function

All can be chained and should be highly compatible with most things.

The order matters and depends on your needs.

The best chaining order is therefore to be determined by your own preferences.

They can be used multiple times in the same workflow!

All are to be used like any model patching node, right after the model loader.

# Nodes:

## Pre CFG automatic scale

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/0437bf5e-1864-41ce-b929-654612b648a6)

### mode:
- Automatic CFG:  applies the same predictable scaling as my other nodes based on this logic
- Strict scaling: applies a scaling which will always give the exact desired value. This tends to create artifacts and random blurs if carried through the end.

### Support empty uncond:

If you use the already available node named ConditioningSetTimestepRange you can stop generating a negative prediction earlier by letting your negative conditioning go through it while setting it like this:

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/4bb39087-d02a-4dd9-821d-dc1f43870eb0)

This speeds up your generation speed by two for the steps where there is no negative.

The only issue if you do this is that the CFG function will weight your positive prediction times your CFG scale against nothing and you will get a black image.

"support_empty_uncond" therefore divides your positive prediction by your CFG scale and avoids this issue.

Doing this combination is similar to the "boost" feature of my original automatic CFG node. It can also let you avoid artifacts if you want to use the strict scaling.

## Pre CFG perp-neg

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/606b2ff3-fb81-4964-8e6d-cee97011a623)

Applies the already known [perp-neg logic](https://perp-neg.github.io/).

Code taken and adapted from ComfyAnon implementation.

## Pre CFG sharpening (experimental)

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/ffca8fae-34b0-44fa-bcd5-dc2ed2c625ca)

Subtract from the current step something from the previous step. This tends to make the images sharper and less saturated.

A negative value can be set.

## Pre CFG consensus sharpening (experimental)

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/af313fe0-f4d5-42dc-ac1c-07c1b84ef96c)

Using a custom algorithm based on euclidean distances, taking all steps into consideration.

Subtract from the current step something from the result of the custom function. This tends to make the images sharper and less saturated too.

A negative value can be set.

## Pre CFG exponentiation (experimental)

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/34367216-eccf-411e-8fab-c63ff0f24331)

A value lower than one will simplify the end result and enhance the saturation / contrasts.

A value higher than one will do the opposite and if pushed too far will most likely make a mess.

# Pro tip:

Did you know that my first activity is to write model merging functions?

While the code is too much of a mess to be shared, I do expose my shared models on this [gallery](https://github.com/Extraltodeus/shared_models_galleries)!

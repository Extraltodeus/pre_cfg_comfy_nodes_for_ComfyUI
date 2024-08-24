
# Pre CFG nodes

A set of nodes to prepare the noise predictions before the CFG function

All can be **chained and repeated** within the same workflow!

They are to be highly compatible with most nodes.

The order matters and depends on your needs.

The best chaining order is therefore to be determined by your own preferences.

All are to be used like any model patching node, right after the model loader.

# Nodes:

## Gradient scaling:

Named like this because I initially wanted to test what would happen if I used, instead of a single CFG scale, a tensor shaped like the latent space with a gradual variation. So, not the kind of gradient used for backpropagation. Then why not try to use masks instead? And what if I could make it so each value will participate so the image would match as close as possible to an input image?

The result is an arithmetic scaling method which does not noticeably slow down the sampling while also scaling the intensity of the values like an "automatic cfg".

So here it is:

![image](https://github.com/user-attachments/assets/86e52c18-d85b-47cc-aee7-cf8750e50bb2)

So, simply put:

- Maximum scale: Which max CFG scale can be used to try to match the input? You can go as high as 500 and still get an output. At 1000 you should stop before the end.
- Minimum scale: Same of course but this one I find better to let in between 3.5 and 5.
- Strength: An overall multiplier for the effect. Generally left at 1 but if you use a plain color image and feel like your results are too smooth you may want to lower it.
- end at sigma: You can go down to the end of the sampling if using the next described toggle but in general I prefer to stop at 0.28. Stopping before the end will give better result with super high scales. 0.28 is the default value.
- Converging scales: make the min and max scales join your sampler scale as the sampling goes. This can weaken the pattern matching effect if you are aiming for something precise but otherwise greatly enhance the final result also allow the use of a bigger maximum scale.
- invert mask: for convenience

### Potential uses:

General light direction/composition influence (all same seed):

![combined_image](https://github.com/user-attachments/assets/647589b4-cea2-41c9-804f-fc59b7ba1b71)

Vignetting:

![combined_v_images](https://github.com/user-attachments/assets/fd492fad-634f-43ce-9d48-918bc56103a9)

Color influence:

![combined_rgb_image](https://github.com/user-attachments/assets/0e71e294-0d5f-4ab8-89ca-1012bc2528df)

Pattern matching, here with a black and white spiral:

![00347UI_00001_](https://github.com/user-attachments/assets/3b030e29-ba5b-4841-bbe7-eb5ae59d652c)

A blue one with a lower scale:

![00297UI_00001_](https://github.com/user-attachments/assets/bc271aa5-93d3-4438-8600-20ae05d47df3)

As you can notice the details a pretty well done in general. It seems that using an input latent as a guide also helps with the overall quality. Using a "freshly" encoded latent, I haven't tried to loop back a latent space resulting from sampling directly.

Text is a bit harder to enforce and may require more tweaking with the scales:

![00133UI_00001_](https://github.com/user-attachments/assets/9c8f1ae3-0411-401f-a6e8-3b4451479576)


Since it takes advantage of the "wiggling room" left by the CFG scale so to make the generation match an image, it can hardly contradict what is being generated.

Here, an example using a black and red spiral, since the base description is about black and white I could only enforce the red by using destructive scales:

![combined_side_by_side_image](https://github.com/user-attachments/assets/f0a85a4b-4ad3-4d20-8248-6d1e81bdddc9)

### Side use:

- If only using a mask for the input, will apply the selected maximum scale to the target area.
- If nothing is connected: will use the positive prediction as guide for 74% of the sigma and the negative for the last part.

Note:

- Given that this is a non-ml solution, unlike controlnet, it can not tell the difference in between a banana and a person. It simply tries to make the values match the input image. A giraffe is just an apple with different values at a different place.
- It is possible to chain multiple times this node for as long as the sum of all the strength sliders is equal or below one.
- I added two image generators. One simply using RGB sliders and a gradient generator which can also make circular patterns while outputting a mask, to make vignetting easy. You will find them in the "image" category.

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

If you want to use this option in a chained setup using this node multiple times I recommand to use it only once and on the last.

## Pre CFG perp-neg

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/606b2ff3-fb81-4964-8e6d-cee97011a623)

Applies the already known [perp-neg logic](https://perp-neg.github.io/).

Code taken and adapted from ComfyAnon implementation.

The context length (added after the screenshot of the node) can be set to a higher value if you are using a tensor rt engine requiring a higher context length.

For more details you can check [my node related to this "Conditioning crop or fill"](https://github.com/Extraltodeus/Uncond-Zero-for-ComfyUI?tab=readme-ov-file#conditioning-crop-or-fill) where I explain a bit more about this.

## Pre CFG sharpening (experimental)

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/ffca8fae-34b0-44fa-bcd5-dc2ed2c625ca)

Subtract from the current step something from the previous step. This tends to make the images sharper and less saturated.

A negative value can be set.

## Pre CFG exponentiation (experimental)

![image](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/assets/15731540/34367216-eccf-411e-8fab-c63ff0f24331)

A value lower than one will simplify the end result and enhance the saturation / contrasts.

A value higher than one will do the opposite and if pushed too far will most likely make a mess.


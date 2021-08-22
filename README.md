# Augmentation
A technique to increase the diversity of training set by applying random (but realistic) transformations.

This repo contains implemenatation of major data augmentations related to vision(image), audio(speech), text(language) in PyTorch 

## Vision
- Horizontal-Vertical flip
- Color jittering: the brightness, contrast, saturation and hue of the image are shifted by a uniformly random
offset applied on all the pixels of the same image. The order in which these shifts are performed is randomly
selected for each patch
- color dropping: an optional conversion to grayscale. When applied, output intensity for a pixel (r, g, b)
corresponds to its luma component, computed as 0.2989r + 0.5870g + 0.1140b;
- Gaussian blurring: for a 224×224 image, a square Gaussian kernel of size 23×23 is used, with a standard
deviation uniformly sampled over [0.1, 2.0];
- solarization: an optional color transformation x 7→ x · 1{x<0.5} + (1 − x)· 1{x≥0.5} for pixels with values
in [0, 1].

## Audio
## Text

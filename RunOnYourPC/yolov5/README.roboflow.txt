
COIN V2 - v1 FlipBlurRB
==============================

This dataset was exported via roboflow.ai on June 20, 2021 at 6:46 PM GMT

It includes 1709 images.
COIN are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random brigthness adjustment of between -25 and +25 percent
* Random Gaussian blur of between 0 and 2 pixels



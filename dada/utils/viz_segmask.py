import numpy as np
from PIL import Image

palette_16 = [128, 64, 128, # road 
            244, 35, 232,   # sidewalk
            70, 70, 70,     # building
            102, 102, 156,  # wall
            190, 153, 153,  # fence
            153, 153, 153,  # pole
            250, 170, 30,   # light
            220, 220, 0,    # sign
            107, 142, 35,   # vegetation
            70, 130, 180,   # sky
            220, 20, 60,    # person
            255, 0, 0,      # rider
            0, 0, 142,      # car
            0, 60, 100,     # bus
            0, 0, 230,      # motocycle
            119, 11, 32]    # bicycle
zero_pad = 256 * 3 - len(palette_16)
for i in range(zero_pad):
    palette_16.append(0)

palette_7 = [128, 64, 128,              # float
            70, 70, 70,                 # construction
            153, 153, 153,              # object
            107, 142, 35,               # nature
            70, 130, 180,               # sky
            220, 20, 60,                # human
            0, 0, 142]                  # vehicle
zero_pad = 256 * 3 - len(palette_7)
for i in range(zero_pad):
    palette_7.append(0)


def colorize_mask(num_classes, mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    if num_classes == 16:
        new_mask.putpalette(palette_16)
    elif num_classes == 7:
        new_mask.putpalette(palette_7)
    return new_mask
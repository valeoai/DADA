# --------------------------------------------------------
# SYNTHIA dataloader
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import numpy as np

from advent.dataset.base_dataset import BaseDataset

from dada.dataset.depth import get_depth


class SYNTHIADataSetDepth(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        set="all",
        num_classes=16,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        use_depth=False,
    ):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        # map to cityscape's ids
        if num_classes == 16:
            self.id_to_trainid = {
                3: 0,
                4: 1,
                2: 2,
                21: 3,
                5: 4,
                7: 5,
                15: 6,
                9: 7,
                6: 8,
                1: 9,
                10: 10,
                17: 11,
                8: 12,
                19: 13,
                12: 14,
                11: 15,
            }
        elif num_classes == 7:
            self.id_to_trainid = {
                1:4, 
                2:1, 
                3:0, 
                4:0, 
                5:1, 
                6:3, 
                7:2, 
                8:6, 
                9:2, 
                10:5, 
                11:6, 
                15:2, 
                22:0}
        else:
            raise NotImplementedError(f"Not yet supported {num_classes} classes")
        # use depth?
        self.use_depth = use_depth
        if self.use_depth:
            for (i, file) in enumerate(self.files):
                img_file, label_file, name = file
                depth_file = self.root / "Depth" / name
                self.files[i] = (img_file, label_file, depth_file, name)
            # disable multi-threading in opencv. Could be ignored
            import os
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"

    def get_metadata(self, name):
        img_file = self.root / "RGB" / name
        label_file = self.root / "parsed_LABELS" / name
        return img_file, label_file

    def __getitem__(self, index):
        if self.use_depth:
            img_file, label_file, depth_file, name = self.files[index]
        else:
            img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        if self.use_depth:
            depth = self.get_depth(depth_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        image = image.copy()
        label_copy = label_copy.copy()
        shape = np.array(image.shape)
        if self.use_depth:
            return image, label_copy, depth.copy(), shape, name
        else:
            return image, label_copy, shape, name

    def get_depth(self, file):
        return get_depth(self, file)

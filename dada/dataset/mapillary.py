import numpy as np
from PIL import Image

from advent.utils import project_root
from advent.utils.serialization import json_load
from advent.dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'dada/dataset/mapillary_list/info.json'
MAPILLARY_2_CITYSCAPES = {'bird': 'other',
                                            'ground animal': 'other',
                                            'curb': 'construction',
                                            'fence': 'construction',
                                            'guard rail': 'construction',
                                            'barrier': 'construction',
                                            'wall': 'construction',
                                            'bike lane': 'flat',
                                            'crosswalk - plain': 'flat',
                                            'curb cut': 'flat',
                                            'parking': 'flat',
                                            'pedestrian area': 'flat',
                                            'rail track': 'flat',
                                            'road': 'flat',
                                            'service lane': 'flat',
                                            'sidewalk': 'flat',
                                            'bridge': 'construction',
                                            'building': 'construction',
                                            'tunnel': 'construction',
                                            'person': 'human',
                                            'bicyclist': 'human',
                                            'motorcyclist': 'human',
                                            'other rider': 'human',
                                            'lane marking - crosswalk': 'flat',
                                            'lane marking - general': 'flat',
                                            'mountain': 'other',
                                            'sand': 'other',
                                            'sky': 'sky',
                                            'snow': 'other',
                                            'terrain': 'flat',
                                            'vegetation': 'nature',
                                            'water': 'other',
                                            'banner': 'other',
                                            'bench': 'other',
                                            'bike rack': 'other',
                                            'billboard': 'other',
                                            'catch basin': 'other',
                                            'cctv camera': 'other',
                                            'fire hydrant': 'other',
                                            'junction box': 'other',
                                            'mailbox': 'other',
                                            'manhole': 'other',
                                            'phone booth': 'other',
                                            'pothole': 'object',
                                            'street light': 'object',
                                            'pole': 'object',
                                            'traffic sign frame': 'object',
                                            'utility pole': 'object',
                                            'traffic light': 'object',
                                            'traffic sign (back)': 'object',
                                            'traffic sign (front)': 'object',
                                            'trash can': 'other',
                                            'bicycle': 'vehicle',
                                            'boat': 'vehicle',
                                            'bus': 'vehicle',
                                            'car': 'vehicle',
                                            'caravan': 'vehicle',
                                            'motorcycle': 'vehicle',
                                            'on rails': 'vehicle',
                                            'other vehicle': 'vehicle',
                                            'trailer': 'vehicle',
                                            'truck': 'vehicle',
                                            'wheeled slow': 'vehicle',
                                            'car mount': 'other',
                                            'ego vehicle': 'other',
                                            'unlabeled': 'other'}
COMMON_7CLASSES = {'flat': 0,
                 'construction': 1,
                 'object': 2,
                 'nature': 3,
                 'sky': 4,
                 'human': 5,
                 'vehicle': 6,
                 'other': 255}

class MapillaryDataSet(BaseDataset):
    def __init__(self, root, list_path, set='train',
                 max_iters=None,
                 crop_size=(321, 321), 
                 mean=(128, 128, 128),
                 info_path=DEFAULT_INFO_PATH, 
                 class_mappings=MAPILLARY_2_CITYSCAPES,
                 class_list=COMMON_7CLASSES,
                 scale_label=True
                 ):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        self.info = json_load(info_path)
        self.map_vector = None
        self.scale_label = scale_label
        ori_class_names = [label['readable'] for label in self.info['labels']]
        if class_mappings is not None:
            self.map_vector = array_from_class_mappings(ori_class_names,
                                                        class_mappings,
                                                        class_list)
        self.class_names = list(class_list.keys())

    def get_metadata(self, name):
        img_file = self.root / self.set / 'images' / name
        label_name = name.replace(".jpg", ".png")
        label_file = self.root / self.set / 'labels' / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = Image.open(img_file).convert('RGB')
        image = resize_with_pad(self.image_size, image, Image.BICUBIC)
        image = self.preprocess(image)
        label = Image.open(label_file)
        if self.scale_label:
            label = resize_with_pad(    self.image_size, 
                                        label,
                                        Image.NEAREST,
                                        fill_value=len(self.class_names) - 1)
        else:
            label = pad_with_fixed_AS(  self.image_size[0]/self.image_size[1],
                                        label,
                                        fill_value=len(self.class_names) - 1)
        if self.map_vector is not None:
            label = self.map_labels(label).copy()

        return image.copy(), label, np.array(image.shape), name
    
def label_mapping_mapilliary(input, mapping):
    output = np.copy(input)
    for ind,val in enumerate(mapping):
        output[input == ind] = val
    return np.array(output, dtype=np.int64)

def array_from_class_mappings(dataset_classes, class_mappings, model_classes):
    """
    :param dataset_classes: list or dict. Mapping between indexes and name of classes.
                            If using a list, it's equivalent
                            to {x: i for i, x in enumerate(dataset_classes)}
    :param class_mappings: Dictionary mapping names of the dataset to
                           names of classes of the model.
    :param model_classes:  list or dict. Same as dataset_classes,
                           but for the model classes.
    :return: A numpy array representing the mapping to be done.
    """
    # Assert all classes are different.
    assert len(model_classes) == len(set(model_classes))

    # to generate the template to fill the dictionary for class_mappings
    # uncomment this code.
    """
    for x in dataset_classes:
        print((' ' * 20) + f'\'{name}\': \'\',')
    """

    # Not case sensitive to make it easier to write.
    if isinstance(dataset_classes, list):
        dataset_classes = {x: i for i, x in enumerate(dataset_classes)}
    dataset_classes = {k.lower(): v for k, v in dataset_classes.items()}
    class_mappings = {k.lower(): v.lower() for k, v in class_mappings.items()}
    if isinstance(model_classes, list):
        model_classes = {x: i for i, x in enumerate(model_classes)}
    model_classes = {k.lower(): v for k, v in model_classes.items()}

    result = np.zeros((max(dataset_classes.values()) + 1,), dtype=np.uint8)
    for dataset_class_name, i in dataset_classes.items():
        result[i] = model_classes[class_mappings[dataset_class_name]]
    return result

def resize_with_pad(target_size, image, resize_type, fill_value=0):
    if target_size is None:
        return np.array(image)
    # find which size to fit to the target size
    target_ratio = target_size[0] / target_size[1]
    image_ratio = image.size[0] / image.size[1]

    if image_ratio > target_ratio:
        resize_ratio = target_size[0] / image.size[0]
        new_image_shape = (target_size[0], int(image.size[1] * resize_ratio))
    else:
        resize_ratio = target_size[1] / image.size[1]
        new_image_shape = (int(image.size[0] * resize_ratio), target_size[1])

    image_resized = image.resize(new_image_shape, resize_type)

    image_resized = np.array(image_resized)
    if image_resized.ndim == 2:
        image_resized = image_resized[:, :, None]

    result = np.ones(target_size[::-1] + [image_resized.shape[2],], np.float32) * fill_value
    assert image_resized.shape[0] <= result.shape[0]
    assert image_resized.shape[1] <= result.shape[1]
    placeholder = result[:image_resized.shape[0], :image_resized.shape[1]]
    placeholder[:] = image_resized
    return result

def pad_with_fixed_AS(target_ratio, image, fill_value=0):
    dimW = float(image.size[0])
    dimH = float(image.size[1])
    image_ratio = dimW/dimH
    if target_ratio > image_ratio:
        dimW = target_ratio*dimH
    elif target_ratio < image_ratio:
        dimH = dimW/target_ratio
    else:
        return np.array(image)
    image = np.array(image)
    result = np.ones((int(dimH), int(dimW)), np.float32) * fill_value
    placeholder = result[:image.shape[0], :image.shape[1]]
    placeholder[:] = image
    return result
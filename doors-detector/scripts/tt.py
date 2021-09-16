import os
from typing import NoReturn

import cv2
import numpy
import numpy as np
import generic_dataset.utilities.save_load_methods as slm
from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.generic_sample import synchronize_on_fields
from generic_dataset.sample_generator import SampleGenerator
COLORS = [(0, 0, 255), (0, 255, 0)]

@synchronize_on_fields(field_names={'bgr_image', 'depth_image', 'bounding_boxes'}, check_pipeline=True)
def visualize(self) -> NoReturn:
    """
    This method visualizes the sample, showing all its fields.
    :return:
    """
    bgr_image = self.get_bgr_image()
    depth_image = self.get_depth_image()
    img_bounding_boxes = bgr_image.copy()

    for label, *box in self.get_bounding_boxes():
        cv2.rectangle(img_bounding_boxes, box, color=COLORS[label], thickness=1)

    row_1 = np.concatenate((bgr_image, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)), axis=1)
    row_1 = np.concatenate((row_1, img_bounding_boxes), axis=1)

    cv2.imshow('Sample', row_1)
    cv2.waitKey()


# The bounding_boxes field is a numpy array of list [[label, x1, y1, width, height]],
# where label is the bounding box label and (x1, y1) are the coordinates of the top left point and width height the bbox dimension

DOOR_LABELS = {0: 'Closed door', 1: 'Opened door'}

DoorSample = SampleGenerator(name='DoorSample', label_set={0, 1}) \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray, save_function=slm.save_cv2_image_bgr, load_function=slm.load_cv2_image_bgr) \
    .add_dataset_field(field_name='depth_image', field_type=np.ndarray, save_function=slm.save_cv2_image_bgr, load_function=slm.load_cv2_image_grayscale) \
    .add_dataset_field(field_name='bounding_boxes', field_type=np.ndarray, default_value=np.array([]), load_function=slm.load_compressed_numpy_array, save_function=slm.save_compressed_numpy_array) \
    .add_custom_method(method_name='visualize', function=visualize) \
    .generate_sample_class()
old_path = '/media/antonazzi/hdd/doors_dataset_small/house7/0/depth_data'
new_path = '/media/antonazzi/hdd/doors_dataset_small/house7/0/depth_image'

def round(data, engine):
    data[data > 10.0] = 10.0
    return data, engine
"""
for file in os.listdir(old_path):
    f = file[:-7]
    pipeline_generate_depth_image = DataPipeline().add_operation(operation=round) \
        .add_operation(round).add_operation(lambda data, engine: ((data * (255.0 / 10.0)).astype(engine.uint8), engine))

    depth_data = slm.load_compressed_numpy_array(os.path.join(old_path, f))
    depth_image = pipeline_generate_depth_image.set_data(depth_data).set_end_function(lambda x: (x)).run(use_gpu=False).get_data()

    

    slm.save_cv2_image_bgr(os.path.join(new_path, 'depth_image' + f[10:]), depth_image)
"""

path = '/home/michele/myfiles/final_doors_dataset/house10/1/bounding_boxes'

for file in os.listdir(path):
    f = file[:-7]
    bounding_boxes = slm.load_compressed_numpy_array(os.path.join(path, f))
    new_bboxes = []
    for l, *x in bounding_boxes:
        if l > 1:
            print('SBAGLIATO')
            l = 1
        new_bboxes.append([l, *x])
    slm.save_compressed_numpy_array(path=os.path.join(path, f), data=np.array(new_bboxes, dtype=int))


#DatasetFolderManager(dataset_path='/home/michele/myfiles/final_doors_dataset', folder_name='house9', sample_class=DoorSample).save_metadata()
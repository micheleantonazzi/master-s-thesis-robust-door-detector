from abc import abstractmethod
from typing import Type, List
from PIL import Image
import numpy as np
import torch
from gibson_env_utilities.doors_dataset.door_sample import DoorSample

import doors_detector.utilities.transforms as T
import pandas as pd
from torch.utils.data import Dataset

SET = Type[str]
TRAIN_SET: SET = 'train_set'
TEST_SET: SET = 'test_set'

DATASET = Type[str]
GIBSON_DATASET: DATASET = 'gibson_dataset'
DEEP_DOORS_2: DATASET = 'deep_doors_2'


class TorchDataset(Dataset):
    def __init__(self, dataset_path: str, dataframe: pd.DataFrame, set_type: SET, std_size: int, max_size: int, scales: List[int]):
        self._dataset_path = dataset_path
        self._dataframe = dataframe
        self._set_type = set_type

        if set == TEST_SET:
            self._transform = T.Compose([
                T.RandomResize([std_size], max_size=max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:

            self._transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(200, 400),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def get_dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def __len__(self):
        return len(self._dataframe.index)

    @abstractmethod
    def load_sample(self, idx) -> DoorSample:
        pass

    def __getitem__(self, idx):
        door_sample = self.load_sample(idx)

        target = {}
        (h, w, _) = door_sample.get_bgr_image().shape
        target['size'] = torch.tensor([int(h), int(w)], dtype=torch.int)

        # Normalize bboxes' size. The bboxes are initially defined as (x_top_left, y_top_left, width, height)
        # Bboxes representation changes, becoming a tuple (center_x, center_y, width, height).
        # All values must be normalized in [0, 1], relative to the image's size
        boxes = door_sample.get_bounding_boxes()
        boxes = np.array([(x, y, x + w, y + h) for label, x, y, w, h in boxes])
        #bboxes = boxes / [(w, h, w, h) for _ in range(len(boxes))]

        target['boxes'] = torch.tensor(boxes, dtype=torch.float)
        target['labels'] = torch.tensor([label for label, *box in door_sample.get_bounding_boxes()], dtype=torch.long)

        # The BGR image is convert in RGB
        img, target = self._transform(Image.fromarray(door_sample.get_bgr_image()), target)

        return img, target, door_sample
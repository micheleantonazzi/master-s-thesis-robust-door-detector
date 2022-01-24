from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from generic_dataset.dataset_manager import DatasetManager

from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample
from doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal
from doors_detector.dataset.torch_dataset import TorchDataset, TRAIN_SET, SET, TEST_SET


class TorchDatasetModelOutput(DatasetDoorsFinal):
    def __init__(self, dataset_path: str,
                 dataframe: pd.DataFrame,
                 targets: List,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):


        super(TorchDatasetModelOutput, self).__init__(
            dataset_path=dataset_path,
            dataframe=dataframe,
            set_type=set_type,
            std_size=std_size,
            max_size=max_size,
            scales=scales)
        self._targets = targets


    def __getitem__(self, idx):

        door_sample, folder_name, absolute_count = super().load_sample(idx)
        target = {}
        (h, w, _) = door_sample.get_bgr_image().shape
        target['size'] = torch.tensor([int(h), int(w)], dtype=torch.int)
        bboxes = np.array([(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for (x, y, w, h) in self._targets[idx]['bboxes']])
        target['boxes'] = torch.tensor(bboxes * [w, h, w, h], dtype=torch.float)
        target['labels'] = torch.tensor(self._targets[idx]['labels'], dtype=torch.long)
        target['folder_name'] = folder_name
        target['absolute_count'] = absolute_count

        # The BGR image is convert in RGB
        img, target = self._transform(Image.fromarray(door_sample.get_bgr_image()[..., [2, 1, 0]]), target)

        return img, target, door_sample


class DatasetCreatorFineTuneModelOutput:
    def __init__(self, dataset_path: str, folder_name: str, test_dataset: DatasetDoorsFinal):
        self._dataset_path = dataset_path
        self._folder_name = folder_name
        self._test_dataset = test_dataset
        self._absolute_counts: list = []
        self._targets = []

    def add_train_sample(self, absolute_counts: int, targets):
        self._absolute_counts.append(absolute_counts)
        self._targets.append(targets)

    def create_datasets(self):
        print([[self._folder_name for _ in range(len(self._absolute_counts))], self._absolute_counts])
        dataframe_train = pd.DataFrame(data={
            'folder_name': [self._folder_name for _ in range(len(self._absolute_counts))],
            'folder_absolute_count': self._absolute_counts
        })
        return (TorchDatasetModelOutput(dataset_path=self._dataset_path, dataframe=dataframe_train, targets=self._targets, set_type=TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                self._test_dataset)



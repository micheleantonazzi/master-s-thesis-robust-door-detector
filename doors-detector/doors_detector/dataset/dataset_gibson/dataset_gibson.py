import numpy as np
import pandas as pd
import torch
from generic_dataset.dataset_manager import DatasetManager
from generic_dataset.utilities.color import Color
from gibson_env_utilities.doors_dataset.door_sample import DoorSample
import doors_detector.utilities.transforms as T
from PIL import Image
from typing import Type, List
from doors_detector.dataset.torch_dataset import TorchDataset, SET, TRAIN_SET, TEST_SET


class DatasetGibson(TorchDataset):
    def __init__(self, dataset_path: str,
                 dataframe: pd.DataFrame,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):

        super(DatasetGibson, self).__init__(
            dataset_path=dataset_path,
            dataframe=dataframe,
            set_type=set_type,
            std_size=std_size,
            max_size=max_size,
            scales=scales)

        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)

    def load_sample(self, idx) -> DoorSample:
        row = self._dataframe.iloc[idx]
        folder_name, absolute_count = row.folder_name, row.folder_absolute_count

        door_sample: DoorSample = self._doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        door_sample.set_pretty_semantic_image(door_sample.get_semantic_image().copy())
        door_sample.create_pretty_semantic_image(color=Color(red=0, green=255, blue=0))
        door_sample.pipeline_depth_data_to_image().run(use_gpu=False).get_data()

        return door_sample


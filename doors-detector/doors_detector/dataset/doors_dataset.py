import numpy as np
import pandas as pd
import torch
from generic_dataset.dataset_manager import DatasetManager
from generic_dataset.utilities.color import Color
from gibson_env_utilities.doors_dataset.door_sample import DoorSample
from torch.utils.data import Dataset
import torchvision.transforms as T


class DoorsDataset(Dataset):
    def __init__(self, dataset_path, dataframe: pd.DataFrame):
        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = dataframe
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self._dataframe.index)

    def __getitem__(self, idx):
        row = self._dataframe.iloc[idx]
        folder_name, absolute_count = row.folder_name, row.folder_absolute_count

        door_sample: DoorSample = self._doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        door_sample.set_pretty_semantic_image(door_sample.get_semantic_image().copy())
        door_sample.create_pretty_semantic_image(color=Color(red=0, green=255, blue=0))
        door_sample.pipeline_depth_data_to_image().run(use_gpu=False).get_data()

        target = {}
        (h, w, _) = door_sample.get_bgr_image().shape
        target['size'] = torch.as_tensor([int(h), int(w)])

        # Normalize bboxes' size. The bboxes are initially defined as (x_top_left, y_top_left, width, height)
        # Bboxes representation changes, becoming a tuple (center_x, center_y, width, height).
        # All values must be normalized in [0, 1], relative to the image's size
        boxes = door_sample.get_bboxes()
        boxes = np.array([(x + 0.5 * w, y + 0.5 * h, w, h) for x, y, w, h in boxes])
        bboxes = boxes / [(w, h, w, h) for _ in range(len(boxes))]

        target['boxes'] = torch.tensor(bboxes)
        return self._transform(door_sample.get_bgr_image()), target, door_sample



import os
from typing import Union

import cv2
import numpy as np
import pandas as pd
import generic_dataset.utilities.save_load_methods as slm
from gibson_env_utilities.doors_dataset.door_sample import DoorSample
from sklearn.model_selection import train_test_split

from doors_detector.dataset.dataset_deep_doors_2.dataset_deep_doors_2 import DatasetDeepDoors2
from doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET


class DatasetCreatorDeepDoors2:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path

        file_names = sorted(list(os.listdir(os.path.join(self._dataset_path, 'door_detection', 'Images'))))

        self._dataframe = pd.DataFrame(columns=['file_name', 'label'])
        self._dataframe.file_name = np.array(file_names)

        # Collect labels from classification folder
        for set, folder, label in [(set, folder, label) for set in ['test', 'train', 'val'] for folder, label in [('Closed', 0), ('Open', 2), ('Semi', 1)]]:
            file_names = os.listdir(os.path.join(self._dataset_path, 'door_classification', 'RGB', set, folder))
            self._dataframe.label[self._dataframe.file_name.isin(file_names)] = label

        self._dataframe.dropna(subset=['label'], inplace=True)

        """
        samples = [DoorSample(label=1) for _ in range(len(self._dataframe.index))]
        
        
        for sample, file_name, label in zip(samples, self._dataframe.file_name, self._dataframe.label):
            sample.set_bgr_image(
                cv2.imread(
                    os.path.join(self._dataset_path, 'door_detection', 'Images', file_name)
                )
            )

            sample.set_pretty_semantic_image(
                cv2.imread(
                    os.path.join(self._dataset_path, 'door_detection', 'Annotations', file_name)
                )
            )

            bboxes = sample.get_bboxes_from_semantic_image()

            sample.set_bounding_boxes(np.array([(label, *rect) for rect in bboxes]))

        self._dataframe.door_sample = np.array(samples)
        """

    def get_label(self):
        return {0: 'Closed door', 1: 'Semi opened door', 2: 'Opened door'}

    def creates_dataset(self, train_size: Union[float, int],
                        test_size: Union[float, int],
                        random_state: int = 42):

        if isinstance(train_size, float):
            assert 0.0 <= train_size <= 1.0

        if isinstance(test_size, float):
            assert 0.0 <= test_size <= 1.0

        assert isinstance(test_size, float) and isinstance(train_size, float) or isinstance(test_size, int) and isinstance(train_size, int)

        train, test = train_test_split(self._dataframe.index.tolist(), train_size=train_size, test_size=test_size, random_state=random_state)

        train_dataframe = self._dataframe.loc[train]
        test_dataframe = self._dataframe.loc[test]

        def print_information(dataframe):
            print(f'    - total samples = {len(dataframe.index)}\n'
                  f'    - Labels considered: {sorted(dataframe.label.unique())}\n'
                  f'    - Total samples in folder: ')
            for label in sorted(dataframe.label.unique()):
                print(f'        - {label}: {len(dataframe[dataframe.label == label])} samples')

        for m, d in zip(['Datasets summary:', 'Train set summary:', 'Test set summary:'], [self._dataframe, train_dataframe, test_dataframe]):
            print(m)
            print_information(d)

        return (DatasetDeepDoors2(dataset_path=self._dataset_path, dataframe=train_dataframe, set_type=TRAIN_SET, std_size=700, max_size=1500, scales=[640 + i * 32 for i in range(11)]), \
               DatasetDeepDoors2(dataset_path=self._dataset_path, dataframe=test_dataframe, set_type=TEST_SET, std_size=700, max_size=1500, scales=[640 + i * 32 for i in range(11)]))



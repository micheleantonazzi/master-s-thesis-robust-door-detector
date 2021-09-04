from typing import Union

from sklearn.utils import shuffle

from doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET, SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample, DOOR_LABELS
import numpy as np
from sklearn.model_selection import train_test_split

from doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal


class DatasetsCreatorDoorsFinal:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()

        self._experiment = 1
        self._folder_name = None

    def get_labels(self):
        return DOOR_LABELS

    def consider_samples_with_label(self, label: int) -> 'DatasetsCreatorGibson':
        """
        This method sets the class to consider only the samples with the given label.
        Other samples (with different labels) are not considered in the datasets' creations
        :param label: the label of the samples to include in the dataset
        :return: the instance of DatasetsCreatorDoorsFinal
        """
        self._dataframe = self._dataframe[self._dataframe.label == label]
        return self

    def set_experiment_number(self, experiment: int, folder_name: str) -> 'DatasetsCreatorDoorsFinal':
        """
        This method is used to set up the experiment to run.
        This first experiment involves training the model using k-1 folders and
        testing it with all the examples in the remaining folder.
        The second experiment involves fine-tuning the previously trained model using some examples of the test data used in experiment 1.
        This new training data belongs to a new environment, never seen in the first training phase. The remaining sample of the k-th folder are used as a test set.
        :param experiment: the number of the experiment to perform
        :param folder_name: the name of the folder to use as a test set in experiment 1 or to split into training a test sets in experiment 2.
        :return: the instance of DatasetsCreatorDoorsFinal
        """
        self._experiment = experiment
        self._folder_name = folder_name

    def creates_dataset(self, train_size: float = 0.1, random_state: int = 42) -> DatasetDoorsFinal:
        """
        This method returns the training and test sets.
        :param train_size: the size of the training set in experiment 2. For the first experiment this parameter is not considered, all samples of folders k-1 are considered.
        """
        if isinstance(train_size, float):
            assert 0.0 < train_size < 1.0

        if self._experiment == 1:
            shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
            train_dataframe = self._dataframe[shuffled_dataframe.folder_name != self._folder_name]
            test_dataframe = self._dataframe[shuffled_dataframe.folder_name == self._folder_name]

        def print_information(dataframe):
            print(f'    - total samples = {len(dataframe.index)}\n'
                  f'    - Folders considered: {sorted(dataframe.folder_name.unique())}\n'
                  f'    - Labels considered: {sorted(dataframe.label.unique())}\n'
                  f'    - Total samples in folder: ')
            for folder in sorted(dataframe.folder_name.unique()):
                print(f'        - {folder}: {len(dataframe[dataframe.folder_name == folder])} samples')
                if DoorSample.GET_LABEL_SET():
                    print(f'        Samples per label:')
                    for label in sorted(list(DoorSample.GET_LABEL_SET())):
                        print(f'            - {label}: {len(dataframe[(dataframe.folder_name == folder) & (dataframe.label == label)])}')
            print()

        for m, d in zip(['Datasets summary:', 'Train set summary:', 'Test set summary:'], [self._dataframe, train_dataframe, test_dataframe]):
            print(m)
            print_information(d)


        return (DatasetDoorsFinal(self._dataset_path, train_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))
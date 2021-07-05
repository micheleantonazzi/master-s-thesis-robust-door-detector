from generic_dataset.dataset_manager import DatasetManager
from gibson_env_utilities.doors_dataset.door_sample import DoorSample
import numpy as np


class DatasetsCreator:
    def __init__(self, dataset_path: str, numpy_seed: int = 0):
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()
        np.random.seed(numpy_seed)

    def consider_samples_with_label(self, label: int) -> 'DatasetsCreator':
        """
        This method sets the class to consider only the samples with the given label.
        Other samples (with different labels) are not considered in the datasets' creations
        :param label: the label of the samples to include in the dataset
        :return: DatasetsCreator itself
        """
        self._dataframe = self._dataframe[self._dataframe.label == label]
        return self

    def consider_n_folders(self, n: int) -> 'DatasetsCreator':
        """
        Sets the DatasetsCreator to consider a fixed number of folder, randomly chosen.
        :raise IndexError if the folders to considers are more than the total number of folders
        :param n: the total number of folders
        :return: DatasetsCreator instance itself
        """
        folders = self._dataset_manager.get_folder_names()
        if n > len(folders):
            raise IndexError(f'You can not consider {n} folders: they are {len(folders)} in total!!')
        selected_folders = np.array(folders)[np.random.choice(len(folders), size=n, replace=False)]
        self._dataframe = self._dataframe[self._dataframe.folder_name.isin(selected_folders)]
        return self

    def creates_dataset(self, sample_counts: int):
        """
        This method returns the training and test sets.
        :param sample_counts: the max num of sample
        :return:
        """
from generic_dataset.dataset_manager import DatasetManager
from gibson_env_utilities.doors_dataset.door_sample import DoorSample


class DatasetsCreator:
    def __init__(self, dataset_path: str):
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()

    def consider_samples_with_label(self, label: int) -> 'DatasetsCreator':
        """
        This method sets the class to consider only the samples with the given label.
        Other samples (with different labels) are not considered in the datasets' creations
        :param label: the label of the samples to include in the dataset
        :return: DatasetsCreator itself
        """
        self._dataframe = self._dataframe[self._dataframe.label == 1]
        return self

    def creates_dataset(self):
        """
        This method returns the training and test sets.
        :return:
        """
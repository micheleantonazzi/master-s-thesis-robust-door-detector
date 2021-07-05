from generic_dataset.dataset_manager import DatasetManager
from gibson_env_utilities.doors_dataset.door_sample import DoorSample


class DatasetsCreator:
    def __init__(self, dataset_path: str):
        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._doors_dataset.get_dataframe()
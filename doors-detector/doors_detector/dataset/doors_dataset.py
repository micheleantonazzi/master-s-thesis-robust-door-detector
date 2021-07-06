import pandas as pd
from generic_dataset.dataset_manager import DatasetManager
from gibson_env_utilities.doors_dataset.door_sample import DoorSample
from torch.utils.data import Dataset


class DoorsDataset(Dataset):
    def __init__(self, dataset_path, dataframe: pd.DataFrame):
        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = dataframe
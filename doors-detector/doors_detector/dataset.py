from generic_dataset.dataset_manager import DatasetManager
from gibson_env_utilities.doors_dataset.door_sample import DoorSample
from torch.utils.data import Dataset


class Dataset

class TorchDoorsDataset(Dataset):
    def __init__(self, dataset_path):
        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
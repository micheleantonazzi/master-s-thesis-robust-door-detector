from typing import Union

from generic_dataset.dataset_manager import DatasetManager
from gibson_env_utilities.doors_dataset.door_sample import DoorSample
import numpy as np
from sklearn.model_selection import train_test_split

from doors_detector.dataset.doors_dataset import DoorsDataset


class DatasetsCreator:
    def __init__(self, dataset_path: str, numpy_seed: int = 0):
        self._dataset_path = dataset_path
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
        selected_folders = np.array(folders)[np.random.choice(len(folders), size=n + 1, replace=False)]
        self._dataframe = self._dataframe[self._dataframe.folder_name.isin(selected_folders)]
        return self

    def creates_dataset(self, train_size: Union[float, int],
                        test_size: Union[float, int],
                        split_folder: bool,
                        folder_train_ratio: float = 0.7,
                        use_all_samples: bool = False,
                        random_state: int = 42):
        """
        This method returns the training and test sets.
        :param train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
                            If int, represents the absolute number of train samples.
        :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                            If int, represents the absolute number of test samples.
        :param split_folder: if True, the train and test set and divided according to the dataset folders.
                The folders are divided according to the folder_ratio parameter.
                Samples from the same folder belong to different sets.
                Otherwise, samples from same folder are mixed into train and test sets.
        :param folder_train_ratio: the ratio of the folders divided into train and test sets. (0.7 means that 70% of folders are included in train set).
                Folders are randomly divided.
        :param use_all_samples: if it is True, all samples in the selected folders are used. In this case, train_size and test_size are ignored.
                                This parameter is considered only if split_folder is True, otherwise it is ignored.
        :return:
        """
        if isinstance(train_size, float):
            assert 0.0 <= train_size <= 1.0

        if isinstance(test_size, float):
            assert 0.0 <= test_size <= 1.0

        assert isinstance(test_size, float) and isinstance(train_size, float) or isinstance(test_size, int) and isinstance(train_size, int)

        if not split_folder:
            train, test, _, _ = train_test_split(self._dataframe.index.tolist(), list(zip(self._dataframe.folder_name.to_list(), self._dataframe.label.to_list())), train_size=train_size, test_size=test_size, random_state=random_state)
            print(train, test)
            train_dataframe = self._dataframe.loc[train]
            test_dataframe = self._dataframe.loc[test]
        else:
            folders = self._dataframe.folder_name.unique()
            folders_train, folders_test = train_test_split(folders, train_size=folder_train_ratio, random_state=random_state)
            train_dataframe = self._dataframe[self._dataframe.folder_name.isin(folders_train)]
            test_dataframe = self._dataframe[self._dataframe.folder_name.isin(folders_test)]

            #Shuffle
            train, train1 = train_test_split(train_dataframe.index.tolist(), train_size=0.99)
            test, test1 = train_test_split(test_dataframe.index.tolist(), train_size=0.99)
            train_dataframe = train_dataframe.loc[train + train1]
            test_dataframe = test_dataframe.loc[test + test1]

            if not use_all_samples:
                train_indexes, _ = train_test_split(train_dataframe.index.tolist(), train_size=train_size, random_state=random_state)

                # If the test size if a float, it must refers to the train size, otherwise, if it is an integer, it will not be modified
                if isinstance(test_size, float) and isinstance(train_size, float):
                    test_size = min(round(len(train_indexes) * test_size), len(test_dataframe.index) - 1)
                test_indexes, _ = train_test_split(test_dataframe.index.tolist(), train_size=test_size, random_state=random_state)
                print(train_indexes, test_indexes)
                train_dataframe = train_dataframe.loc[train_indexes]
                test_dataframe = test_dataframe.loc[test_indexes]

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

        return DoorsDataset(self._dataset_path, train_dataframe), DoorsDataset(self._dataset_path, test_dataframe)
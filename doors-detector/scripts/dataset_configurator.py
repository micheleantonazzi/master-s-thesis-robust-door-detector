import numpy as np

from doors_detector.dataset.dataset_deep_doors_2.dataset_creator_deep_doors_2 import DatasetCreatorDeepDoors2
from doors_detector.dataset.dataset_gibson.datasets_creator_gibson import DatasetsCreatorGibson
from doors_detector.dataset.dataset_deep_doors_2_labelled.dataset_deep_door_2_labelled import DatasetDeepDoors2Labelled
from doors_detector.dataset.dataset_deep_doors_2_labelled.datasets_creator_deep_doors_2_labelled import DatasetsCreatorDeepDoors2Labelled

gibson_dataset_path = '/home/michele/myfiles/doors_dataset_small'
deep_doors_2_dataset_path = '/home/michele/myfiles/deep_doors_2'
deep_doors_2_labelled_dataset_path = '/home/michele/myfiles/deep_doors_2_labelled'

COLORS = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=float) / np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]], dtype=float)


def get_my_doors_sets():
    datasets_creator = DatasetsCreatorGibson(gibson_dataset_path)
    datasets_creator.consider_samples_with_label(label=1)
    datasets_creator.consider_n_folders(1)
    train, test = datasets_creator.creates_dataset(train_size=0.8, test_size=0.2, split_folder=False, folder_train_ratio=0.8, use_all_samples=True)
    labels = datasets_creator.get_labels()

    return train, test, labels


def get_deep_doors_2_sets():
    dataset_creator = DatasetCreatorDeepDoors2(dataset_path=deep_doors_2_dataset_path)

    train, test = dataset_creator.creates_dataset(train_size=0.8, test_size=0.2)
    labels = dataset_creator.get_label()

    return train, test, labels


def get_deep_doors_2_labelled_sets():
    dataset_creator = DatasetsCreatorDeepDoors2Labelled(dataset_path=deep_doors_2_labelled_dataset_path)
    dataset_creator.consider_samples_with_label(label=1)
    train, test = dataset_creator.creates_dataset(train_size=0.8, test_size=0.2)
    labels = dataset_creator.get_labels()

    return train, test, labels


def get_final_doors_dataset():
    dataset_creator = DatasetsCreatorDeepDoors2Labelled(dataset_path=final_doors_dataset_path)
    dataset_creator.consider_samples_with_label(label=1)
    train, test = dataset_creator.creates_dataset(train_size=0.8, test_size=0.2)
    labels = dataset_creator.get_labels()

    return train, test, labels
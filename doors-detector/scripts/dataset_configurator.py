import numpy as np
from doors_detector.dataset.dataset_deep_doors_2_labelled.datasets_creator_deep_doors_2_labelled import DatasetsCreatorDeepDoors2Labelled


deep_doors_2_labelled_dataset_path = '/home/michele/myfiles/deep_doors_2_labelled'
final_doors_dataset_path = '/home/antonazzi/myfiles/final_doors_dataset'

COLORS = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=float) / np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]], dtype=float)


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
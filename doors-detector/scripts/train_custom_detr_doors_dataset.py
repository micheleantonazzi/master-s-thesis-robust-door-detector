from doors_detector.dataset.datasets_creator import DatasetsCreator

door_dataset_path = '/home/michele/myfiles/doors_dataset'

if __name__ == '__main__':
    datasets_creator = DatasetsCreator(door_dataset_path, numpy_seed=0)
    datasets_creator.consider_samples_with_label(label=1)
    datasets_creator.consider_n_folders(3)
    train, test = datasets_creator.creates_dataset(train_size=0.7, test_size=0.3, split_folder=True, folder_train_ratio=0.8, use_all_samples=True)

    img, target, door_sample = train[2]
    door_sample.visualize()
    print(img.size(), target)

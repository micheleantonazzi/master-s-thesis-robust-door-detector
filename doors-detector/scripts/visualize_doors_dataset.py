import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detector.dataset.dataset_gibson.datasets_creator_gibson import DatasetsCreatorGibson
from doors_detector.models.detr import PostProcess
from gibson_env_utilities.doors_dataset.door_sample import DOOR_LABELS
from doors_detector.dataset.dataset_deep_doors_2.dataset_creator_deep_doors_2 import DatasetCreatorDeepDoors2

gibson_dataset_path = '/home/michele/myfiles/doors_dataset_labelled'
deep_doors_2_dataset_path = '/home/michele/myfiles/deep_doors_2'

params = {
    'seed': 0
}

COLORS = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=float) / np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]], dtype=float)


def get_my_doors_sets():
    datasets_creator = DatasetsCreatorGibson(gibson_dataset_path)
    datasets_creator.consider_samples_with_label(label=1)
    datasets_creator.consider_n_folders(1)
    train, test = datasets_creator.creates_dataset(train_size=0.9, test_size=0.1, split_folder=False, folder_train_ratio=0.8, use_all_samples=True)
    labels = datasets_creator.get_labels()

    return train, test, labels


def get_deep_doors_2_sets():
    dataset_creator = DatasetCreatorDeepDoors2(dataset_path=deep_doors_2_dataset_path)

    train, test = dataset_creator.creates_dataset(train_size=0.9, test_size=0.1)
    labels = dataset_creator.get_label()

    return train, test, labels



if __name__ == '__main__':

    # Fix seeds
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    train, test, labels = get_my_doors_sets()
    #train, test, labels = get_deep_doors_2_sets()

    post_processor = PostProcess()

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    for i in range(10):
        img, target, door_sample = train[i]

        boxes = target['boxes']
        outputs = {'pred_boxes': boxes.unsqueeze(0),
                   'pred_logits': torch.tensor([[[-0.99, - 0.2, -0.2, -0.2] for _ in range(len(boxes))]], dtype=torch.float32)
                   }

        for i, label in enumerate(target['labels']):
            outputs['pred_logits'][0][i][label] = 0.7

        processed = post_processor(outputs=outputs, target_sizes=torch.tensor([img.size()[1:]]))[0]

        keep = processed['scores'] > 0.01

        # Show image with bboxes

        # Denormalize image tensor and convert to PIL image
        pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.figure(figsize=(16, 10))

        img = T.ToPILImage(mode='RGB')(pil_image).convert("RGB")
        plt.imshow(img)
        ax = plt.gca()
        print(processed['labels'][keep])
        for label, score, (xmin, ymin, xmax, ymax) in zip(processed['labels'][keep], processed['scores'][keep], processed['boxes'][keep]):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=COLORS[label], linewidth=3))
            text = f'{labels[int(label)]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.show()






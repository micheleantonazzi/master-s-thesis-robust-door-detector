import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detector.dataset.datasets_creator import DatasetsCreator
from doors_detector.models.detr import PostProcess

door_dataset_path = '/home/michele/myfiles/doors_dataset'


params = {
    'seed': 0
}


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

if __name__ == '__main__':

    # Fix seeds
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    datasets_creator = DatasetsCreator(door_dataset_path)
    datasets_creator.consider_samples_with_label(label=1)
    datasets_creator.consider_n_folders(3)
    train, test = datasets_creator.creates_dataset(train_size=0.9, test_size=0.1, split_folder=False, folder_train_ratio=0.8, use_all_samples=True)

    post_processor = PostProcess()

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    for i in range(10):
        img, target, door_sample = train[i]

        boxes = target['boxes']
        outputs = {'pred_boxes': boxes.unsqueeze(0),
                   'pred_logits': torch.tensor([[[-0.99, 0.99] for _ in range(len(boxes))]], dtype=torch.float32)
                   }

        processed = post_processor(outputs=outputs, target_sizes=torch.tensor([img.size()[1:]]))[0]

        keep = processed['scores'] > 0.01

        # Show image with bboxes

        # Denormalize image tensor and convert to PIL image
        pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.figure(figsize=(16, 10))

        plt.imshow(T.ToPILImage(mode='RGB')(pil_image).convert("RGB"))
        ax = plt.gca()

        colors = COLORS * 100
        for label, score, (xmin, ymin, xmax, ymax), c in zip(processed['labels'][keep], processed['scores'][keep], processed['boxes'][keep], colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            text = f'Door: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.show()

        door_sample.visualize()






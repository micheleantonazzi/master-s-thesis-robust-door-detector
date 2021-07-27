import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detector.dataset.dataset_gibson.datasets_creator_gibson import DatasetsCreatorGibson
from doors_detector.dataset.torch_dataset import DEEP_DOORS_2
from doors_detector.models.detr import PostProcess
from doors_detector.models.detr_door_detector import *
from doors_detector.models.model_names import DETR_RESNET50
from scripts.dataset_configurator import *

params = {
    'seed': 0
}


def seedseed_everything(param):
    pass


if __name__ == '__main__':

    # Fix seeds
    seedseed_everything(params['seed'])

    train, test, labels = get_deep_doors_2_sets()

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=True, dataset_name=DEEP_DOORS_2, description=PRETRAINED_FREEZEMODEL_CLASS_BBOX)
    model.eval()

    for i in range(10, 50):
        img, target, door_sample = test[i]
        img = img.unsqueeze(0)
        outputs = model(img)

        """
        # Print real boxes
        outputs['pred_logits'] = torch.tensor([[[0, 1.0] for b in target['boxes']]], dtype=torch.float32)
        outputs['pred_boxes'] = target['boxes'].unsqueeze(0)
        """

        post_processor = PostProcess()
        img_size = list(img.size()[2:])
        processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([img_size]))

        for image_data in processed_data:
            # keep only predictions with 0.7+ confidence

            keep = image_data['scores'] > 0.7

            # Show image with bboxes

            # Denormalize image tensor and convert to PIL image
            pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            plt.figure(figsize=(16, 10))

            plt.imshow(T.ToPILImage(mode='RGB')(pil_image[0]).convert("RGB"))
            ax = plt.gca()

            for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color=COLORS[label], linewidth=3))
                text = f'{labels[int(label)]}: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))

            plt.axis('off')
            plt.show()
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detector.dataset.dataset_gibson.datasets_creator_gibson import DatasetsCreatorGibson
from doors_detector.models.detr import PostProcess
from doors_detector.models.detr_door_detector import DetrDoorDetector
from doors_detector.models.model_names import DETR_RESNET50


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


door_dataset_path = '/home/michele/myfiles/doors_dataset_labelled'

params = {
    'seed': 0
}

if __name__ == '__main__':

    # Fix seeds
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    datasets_creator = DatasetsCreatorGibson(door_dataset_path)
    datasets_creator.consider_samples_with_label(label=1)
    datasets_creator.consider_n_folders(1)
    train, test = datasets_creator.creates_dataset(train_size=0.9, test_size=0.1, split_folder=False, folder_train_ratio=0.8, use_all_samples=True)

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    img, target, door_sample = train[0]

    model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=True)
    model.eval()

    for i in range(0, 10):
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

            keep = image_data['scores'] > 0.6

            # Show image with bboxes

            # Denormalize image tensor and convert to PIL image
            pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            plt.figure(figsize=(16, 10))

            plt.imshow(T.ToPILImage(mode='RGB')(pil_image[0]).convert("RGB"))
            ax = plt.gca()

            colors = COLORS * torch.count_nonzero(keep)
            for label, score, (xmin, ymin, xmax, ymax), c in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep], colors):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color=c, linewidth=3))
                text = f'Door: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))

            plt.axis('off')
            plt.show()
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from doors_detector.dataset.dataset_doors_final.datasets_creator_door_fine_tune_model_output import \
    DatasetCreatorFineTuneModelOutput
from doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detector.models.detr_door_detector import *
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import seed_everything, collate_fn
from scripts.dataset_configurator import get_final_doors_dataset_door_no_door_task
import torch.nn.functional as F
import torchvision.transforms as T


params = {
    'seed': 0,
    'batch_size': 1
}

folder_name = 'house1'
threshold = 0.9
# Fix seeds
seed_everything(params['seed'])

train, test, labels_set, COLORS = get_final_doors_dataset_door_no_door_task(folder_name=folder_name, train_size=0.015, test_size=0.25)

model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels_set.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1)
model.eval()

data_loader_classify = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)

dataset_model_output = DatasetCreatorFineTuneModelOutput(
    dataset_path='/home/michele/myfiles/final_doors_dataset',
    folder_name=folder_name,
    test_dataset=test
)

targets_saved = []

for i, (images, targets) in enumerate(data_loader_classify):

    outputs = model(images)
    pred_logits, pred_boxes_images = outputs['pred_logits'], outputs['pred_boxes']
    prob = F.softmax(pred_logits, -1)

    scores_images, labels_images = prob[..., :-1].max(-1)

    for i, (scores, labels, bboxes) in enumerate(zip(scores_images, labels_images, pred_boxes_images)):
        keep = scores > threshold

        if torch.count_nonzero(keep).item() > 0:
            scores = scores[keep]
            labels = labels[keep]
            bboxes = bboxes[keep]

            #targets_saved.append({
               # 'folder_name': targets[i]['folder_name'],
               # 'absolute_count': targets[i]['absolute_count'],
                #'bboxes': bboxes.tolist(),
                #'labels': labels.tolist()
            #})
            print(targets[i]['absolute_count'], bboxes.tolist())
            dataset_model_output.add_train_sample(targets[i]['absolute_count'], targets={'bboxes': bboxes.tolist(), 'labels': labels.tolist()})
            pil_image = images[i] * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            plt.figure(figsize=(16, 10))

            plt.imshow(T.ToPILImage()(pil_image))
            ax = plt.gca()

            for label, (x, y, w, h), score in zip(labels.tolist(), bboxes.tolist(), scores.tolist()):
                ax.add_patch(plt.Rectangle(((x - w / 2 )*256, (y - h / 2 )*256), w*256, h *256,
                                           fill=False, color=COLORS[label], linewidth=3))
                text = f'{labels_set[int(label)]}: {score:0.2f}'
                ax.text((x - w / 2 )*256, (y - h / 2 )*256, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))

            plt.axis('off')
            plt.show()

train, test = dataset_model_output.create_datasets()

data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)

for i, (imgs, targets) in enumerate(data_loader_train):
    for img, target in zip(imgs, targets):

        # Denormalize image tensor and convert to PIL image
        pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.figure(figsize=(16, 10))

        plt.imshow(T.ToPILImage()(pil_image))
        ax = plt.gca()

        for label, (x, y, w, h) in zip(target['labels'], target['boxes']):

            ax.add_patch(plt.Rectangle(((x - w / 2 )*256, (y - h / 2 )*256), w*256, h*256,
                                       fill=False, color=COLORS[label.item()], linewidth=3))
            text = f'{labels_set[int(label.item())]}: {1:0.2f}'
            ax.text((x - w / 2 )*256, (y - h / 2 )*256, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.show()













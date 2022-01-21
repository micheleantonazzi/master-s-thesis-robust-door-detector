from torch.utils.data import DataLoader

from doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detector.models.detr_door_detector import *
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import seed_everything, collate_fn
from scripts.dataset_configurator import get_final_doors_dataset_door_no_door_task
import torch.nn.functional as F


params = {
    'seed': 0,
    'batch_size': 1
}

threshold = 0.8
# Fix seeds
seed_everything(params['seed'])

train, test, labels, COLORS = get_final_doors_dataset_door_no_door_task(folder_name='house1', train_size=0.25, test_size=0.25)

model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1)
model.eval()

data_loader_classify = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)

for i, (images, targets) in enumerate(data_loader_classify):

    outputs = model(images)
    pred_logits, pred_boxes_images = outputs['pred_logits'], outputs['pred_boxes']
    prob = F.softmax(pred_logits, -1)
    scores_images, labels_images = prob[..., :-1].max(-1)
    print(scores_images)
    print(labels_images)
    keep = scores_images > 0.8

    print(scores_images[keep])
    print(labels_images[keep])
    print(pred_boxes_images[keep])
    print(1)







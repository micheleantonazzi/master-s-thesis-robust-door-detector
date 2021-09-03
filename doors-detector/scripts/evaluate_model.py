from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED
from doors_detector.models.detr import PostProcess
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.coco_evaluator import CocoEvaluator
from doors_detector.utilities.utils import seed_everything, collate_fn
from dataset_configurator import *
from doors_detector.models.detr_door_detector import *


device = 'cuda'
batch_size = 1

if __name__ == '__main__':
    seed_everything(0)

    train, test, labels, _ = get_deep_doors_2_labelled_sets()

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=DEEP_DOORS_2_LABELLED, description=DEEP_DOORS_2_LABELLED_EXP)

    model.eval()
    model.to(device)

    data_loader_test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, drop_last=False, num_workers=4)
    coco_evaluator = CocoEvaluator()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
        images = images.to(device)

        outputs = model(images)

        coco_evaluator.add_predictions(targets=targets, predictions=outputs)

    """img, target, sample = test[0]

    sample.visualize()

    img = img.to(device)

    outputs = model(img.unsqueeze(0))

    

    coco_evaluator.add_predictions(targets=(target,), predictions=outputs)"""
    print(coco_evaluator.get_coco_metrics())

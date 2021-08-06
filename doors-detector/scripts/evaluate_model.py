from doors_detector.dataset.torch_dataset import DEEP_DOORS_2
from doors_detector.models.detr_door_detector import DetrDoorDetector
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import seed_everything
from dataset_configurator import *
from doors_detector.models.detr_door_detector import *


device = 'cuda'

if __name__ == '__main__':
    seed_everything(0)

    train, test, labels = get_deep_doors_2_labelled_sets()

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=True, dataset_name=DEEP_DOORS_2, description=PRETRAINED_FINETUNE_ALL_LR_LOW_STEP_NOAUG_10OBJQUERIES_LABELLED)

    model.eval()
    model.to(device)

    img, target, sample = test[0]

    sample.visualize()
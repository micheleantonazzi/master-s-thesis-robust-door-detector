from doors_detector.dataset.torch_dataset import DEEP_DOORS_2
from doors_detector.models.detr_door_detector import DetrDoorDetector, PRETRAINED_FINETUNE_ALL_LR_LOW_STEP_NOAUG_10OBJQUERIES_LABELLED
from doors_detector.models.model_names import DETR_RESNET50
from scripts.dataset_configurator import get_deep_doors_2_labelled_sets

model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=True, dataset_name=DEEP_DOORS_2, description=PRETRAINED_FINETUNE_ALL_LR_LOW_STEP_NOAUG_10OBJQUERIES_LABELLED)
train, test, labels = get_deep_doors_2_labelled_sets()


img, target, sample = train[0]
sample.visualize()
from doors_detector.dataset.torch_dataset import DEEP_DOORS_2
from doors_detector.models.detr_door_detector import *
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.plot import plot_losses


model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=True, dataset_name=DEEP_DOORS_2, description=PRETRAINED_FINETUNE_ALL_LR_LOW_NOSTEP_NOAUG_10OBJQUERIES_LABELLED)
checkpoint = model.load_checkpoint()

plot_losses(checkpoint['logs'])
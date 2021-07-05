import torch


ModelName = str
DETR_RESNET50: ModelName = 'detr_resnet50'


class Detr:
    def __init__(self, model_name: ModelName, pretrained: bool):
        self._model_name = model_name
        self.model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)


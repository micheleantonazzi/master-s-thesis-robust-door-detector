import os
from typing import Tuple

import torch
from torch import nn

from doors_detector.dataset.torch_dataset import DATASET
from doors_detector.models.mlp import MLP
from doors_detector.models.model_names import ModelName

DESCRIPTION = int

PRETRAINED_FREEZEMODEL_CLASS_BBOX: DESCRIPTION = 1
PRETRAINED_FREEZEMODEL_CLASS_BBOX_EOS05_WEIGHT1 = 2
PRETRAINED_FREEZEMODEL_CLASS: DESCRIPTION = 1

class DetrDoorDetector(nn.Module):
    """
    This class builds a door detector starting from a detr pretrained module.
    Basically it loads a dtr module and modify its structure to recognize door.
    """
    def __init__(self, model_name: ModelName, pretrained: bool, dataset_name: DATASET, description: DESCRIPTION):
        """

        :param model_name: the name of the detr base model
        :param pretrained: it refers to the DetrDoorDetector class, not to detr base model.
                            It True, the DetrDoorDetector's weights are loaded, otherwise the weights are loaded only for the detr base model
        """
        super(DetrDoorDetector, self).__init__()
        self._model_name = model_name
        self.model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)
        self._dataset_name = dataset_name
        self._description = description

        # Freeze the model parameters
        for n, param in self.model.named_parameters():
            if 'bbox_embed' not in n:
                param.requires_grad = False
            else:
                print(n)

        # Change the last part of the model
        #self.model.query_embed = nn.Embedding(10, self.model.transformer.d_model)
        self.model.class_embed = nn.Linear(256, 4)
        #self.model.bbox_embed = MLP(256, 256, 4, 3)

        if pretrained:
            path = os.path.join(os.path.dirname(__file__), 'train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            self.model.class_embed.load_state_dict(torch.load(os.path.join(path, 'class_embed.pth')))
            self.model.bbox_embed.load_state_dict(torch.load(os.path.join(path, 'bbox_embed.pth')))

    def forward(self, x):
        x = self.model(x)

        """
        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape=[batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the un-normalized bounding box.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        return x

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model.to(device)

    def save(self, epoch, optimizer_state_dict, lr_scheduler_state_dict, params, logs):
        path = os.path.join(os.path.dirname(__file__), 'train_params', self._model_name + '_' + str(self._description))

        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, str(self._dataset_name))

        if not os.path.exists(path):
            os.mkdir(path)

        torch.save(self.model.bbox_embed.state_dict(), os.path.join(path, 'bbox_embed.pth'))
        torch.save(self.model.class_embed.state_dict(), os.path.join(path, 'class_embed.pth'))
        torch.save(
            {'epoch': epoch,
             'optimizer_state_dict': optimizer_state_dict,
             'params': params,
             'lr_scheduler_state_dict': lr_scheduler_state_dict,
             'logs': logs}, os.path.join(path, 'checkpoint.pth'))

    def load_checkpoint(self,):
        path = os.path.join(os.path.dirname(__file__), 'train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
        return torch.load(os.path.join(path, 'checkpoint.pth'))




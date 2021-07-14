import time

import torch.optim
from engine import train_one_epoch
from matplotlib import pyplot as plt
from models.detr import SetCriterion
from models.matcher import HungarianMatcher
from torch.utils.data import DataLoader
import torchvision.transforms as T

from doors_detector.dataset.datasets_creator import DatasetsCreator
from doors_detector.models.DetrDoorDetector import DetrDoorDetector
from doors_detector.models.detr import PostProcess
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import collate_fn

door_dataset_path = '/home/michele/myfiles/doors_dataset'

device = 'cuda'

epochs = 1

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Params
lr = 1e-4
weight_decay = 1e-4
lr_drop = 200
batch_size = 10

# Criterion
bbox_loss_coef = 5
giou_loss_coef = 2
eos_coef = 0.1
# Matcher
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2

if __name__ == '__main__':
    datasets_creator = DatasetsCreator(door_dataset_path, numpy_seed=0)
    datasets_creator.consider_samples_with_label(label=1)
    datasets_creator.consider_n_folders(3)
    train, test = datasets_creator.creates_dataset(train_size=0.7, test_size=0.3, split_folder=False, folder_train_ratio=0.8, use_all_samples=True)

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    img, target, door_sample = train[0]

    model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=False)

    print("Params to learn:")
    params_to_optimize = []
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            print('\t', name)


    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)

    # StepLR decays the learning rate of each parameter group by gamma every step_size epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Create criterion to calculate losses
    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
    matcher = HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)
    criterion = SetCriterion(1, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=eos_coef, losses=losses)

    data_loader_train = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)
    data_loader_test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, drop_last=False, num_workers=4)

    model.to(device)
    criterion.to(device)

    start_time = time.time()

    for epoch in range(epochs):
        #train_stats = train_one_epoch(
            #model, criterion, data_loader_train, optimizer, device, epoch,
            #)

        model.train()
        criterion.train()

        for i, training_data in enumerate(data_loader_train):
            # Data is a tuple where
            #   - data[0]: a tensor containing all images shape = [batch_size, channels, img_height, img_width]
            #   - data[1]: a tuple of dictionaries containing the images' targets

            #train_stats = train_one_epoch(
                #model, criterion, data_loader_train, optimizer, device, epoch,
                #)
            
            images, targets = training_data
            images = images.to(device)

            # Move targets to device
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            outputs = model(images)

            # Compute losses
            losses_dict = criterion(outputs, targets)

            # Losses are weighted using parameters contained in a dictionary
            losses = sum(losses_dict[k] * weight_dict[k] for k in losses_dict.keys() if k in weight_dict)

            # Back propagate the losses
            optimizer.zero_grad()

            losses.backward()

            optimizer.step()

            print(f'Epoch [{epoch}] -> [{i}/{len(data_loader_train)}]: ' + str([f'{k}: {v.item()}' for k, v in losses_dict.items()]))


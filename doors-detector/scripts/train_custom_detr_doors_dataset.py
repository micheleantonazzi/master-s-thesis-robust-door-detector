import random
import time

import numpy as np
import torch.optim
from models.detr import SetCriterion
from models.matcher import HungarianMatcher
from torch.utils.data import DataLoader

from doors_detector.dataset.dataset_gibson.datasets_creator_gibson import DatasetsCreatorGibson
from doors_detector.models.detr_door_detector import DetrDoorDetector
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import collate_fn

door_dataset_path = '/home/michele/myfiles/doors_dataset_labelled'

device = 'cuda'


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# Params
params = {
    'epochs': 20,
    'batch_size': 10,
    'seed': 0,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'lr_drop': 200,
    # Criterion
    'bbox_loss_coef': 5,
    'giou_loss_coef': 2,
    'eos_coef': 0.1,
    # Matcher
    'set_cost_class': 1,
    'set_cost_bbox': 5,
    'set_cost_giou': 2,
}

restart_checkpoint = False

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

    model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=restart_checkpoint)
    model.to(device)

    # Loads params if training starts from a checkpoint
    start_epoch = 0
    logs = []
    optimizer_state_dict = {}
    lr_scheduler_state_dict = {}
    if restart_checkpoint:
        checkpoint = model.load_checkpoint()
        start_epoch = checkpoint['epoch'] + 1
        logs = checkpoint['logs']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']

    print("Params to learn:")
    params_to_optimize = []
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            print('\t', name)

    optimizer = torch.optim.AdamW(params_to_optimize, lr=params['lr'], weight_decay=params['weight_decay'])

    # StepLR decays the learning rate of each parameter group by gamma every step_size epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params['lr_drop'])

    if restart_checkpoint:
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    # Create criterion to calculate losses
    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': 1, 'loss_bbox': params['bbox_loss_coef'], 'loss_giou': params['giou_loss_coef']}
    matcher = HungarianMatcher(cost_class=params['set_cost_class'], cost_bbox=params['set_cost_bbox'], cost_giou=params['set_cost_giou'])
    criterion = SetCriterion(1, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=params['eos_coef'], losses=losses)

    data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)
    data_loader_test = DataLoader(test, batch_size=params['batch_size'], collate_fn=collate_fn, drop_last=False, num_workers=4)

    print_logs_every = 10

    criterion.to(device)

    start_time = time.time()

    for epoch in range(start_epoch, params['epochs']):
        #train_stats = train_one_epoch(
            #model, criterion, data_loader_train, optimizer, device, epoch,
        #)

        accumulate_losses = {}

        model.train()
        criterion.train()

        for i, training_data in enumerate(data_loader_train):
            # Data is a tuple where
            #   - data[0]: a tensor containing all images shape = [batch_size, channels, img_height, img_width]
            #   - data[1]: a tuple of dictionaries containing the images' targets
            
            images, targets = training_data
            images = images.to(device)

            # Move targets to device
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            outputs = model(images)

            # Compute losses
            losses_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # Losses are weighted using parameters contained in a dictionary
            losses = sum(losses_dict[k] * weight_dict[k] for k in losses_dict.keys() if k in weight_dict)

            scaled_losses_dict = {k: v * weight_dict[k] for k, v in losses_dict.items() if k in weight_dict}
            scaled_losses_dict['loss'] = sum(scaled_losses_dict.values())

            # Back propagate the losses
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            accumulate_losses = {k: scaled_losses_dict[k] if k not in accumulate_losses else sum([accumulate_losses[k], scaled_losses_dict[k]]) for k, _ in scaled_losses_dict.items()}

            if i % print_logs_every == 0:
                accumulate_losses = {k: v.item() / print_logs_every for k, v in accumulate_losses.items()}
                print(f'Epoch [{epoch}] -> [{i}/{len(data_loader_train)}]: ' + ', '.join([f'{k}: {v}' for k, v in accumulate_losses.items()]))
                logs.append(accumulate_losses)
                accumulate_losses = {}

        lr_scheduler.step()

    model.save(epoch=params['epochs'] - 1,
               optimizer_state_dict=optimizer.state_dict(),
               lr_scheduler_state_dict=lr_scheduler.state_dict(),
               params=params,
               logs=logs
               )


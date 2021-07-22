import random
import time
from doors_detector.dataset.torch_dataset import DEEP_DOORS_2
import numpy as np
import torch.optim
from models.detr import SetCriterion
from models.matcher import HungarianMatcher
from torch.utils.data import DataLoader
from engine import evaluate, train_one_epoch
from doors_detector.dataset.dataset_gibson.datasets_creator_gibson import DatasetsCreatorGibson
from doors_detector.models.detr_door_detector import *
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import collate_fn
from scripts.dataset_configurator import *
from doors_detector.utilities.utils import seed_everything


device = 'cuda'


# Params
params = {
    'epochs': 51,
    'batch_size': 1,
    'seed': 0,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'lr_drop': 200,
    'lr_backbone': 1e-5,
    # Criterion
    'bbox_loss_coef': 5,
    'giou_loss_coef': 2,
    'eos_coef': 0.1,
    # Matcher
    'set_cost_class': 1,
    'set_cost_bbox': 5,
    'set_cost_giou': 2,
}

restart_checkpoint = True

if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    train, test, labels = get_deep_doors_2_sets()

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=restart_checkpoint, dataset_name=DEEP_DOORS_2, description=PRETRAINED_FREEZEMODEL_CLASS)
    model.to(device)

    # Loads params if training starts from a checkpoint
    start_epoch = 0
    logs = {'train': [], 'test': []}
    optimizer_state_dict = {}
    lr_scheduler_state_dict = {}
    if restart_checkpoint:
        checkpoint = model.load_checkpoint()
        start_epoch = checkpoint['epoch'] + 1
        logs = checkpoint['logs']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']

    print("Params to learn:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": params['lr_backbone'],
        },
    ]

    #print(param_dicts)

    optimizer = torch.optim.AdamW(param_dicts, lr=params['lr'], weight_decay=params['weight_decay'])

    # StepLR decays the learning rate of each parameter group by gamma every step_size epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params['lr_drop'])

    if restart_checkpoint:
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    # Create criterion to calculate losses
    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': 1, 'loss_bbox': params['bbox_loss_coef'], 'loss_giou': params['giou_loss_coef']}
    matcher = HungarianMatcher(cost_class=params['set_cost_class'], cost_bbox=params['set_cost_bbox'], cost_giou=params['set_cost_giou'])
    criterion = SetCriterion(3, matcher=matcher, weight_dict=weight_dict,
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

        temp_logs = {'train': [], 'test': []}
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

            if i % print_logs_every == print_logs_every - 1:
                accumulate_losses = {k: v.item() / print_logs_every for k, v in accumulate_losses.items()}
                print(f'Train epoch [{epoch}] -> [{i}/{len(data_loader_train)}]: ' + ', '.join([f'{k}: {v}' for k, v in accumulate_losses.items()]))
                temp_logs['train'].append(accumulate_losses)
                accumulate_losses = {}

        epoch_total = {}
        for d in temp_logs['train']:
            for k in d:
                epoch_total[k] = epoch_total.get(k, 0) + d[k]

        logs['train'].append({k: v / len(temp_logs['train']) for k, v in epoch_total.items()})

        print(f'----> EPOCH SUMMARY TRAIN [{epoch}] -> [{i}/{len(data_loader_train)}]: ' + ', '.join([f'{k}: {v}' for k, v in logs['train'][epoch].items()]))

        with torch.no_grad():
            model.eval()
            criterion.eval()

            accumulate_losses = {}
            for i, test_data in enumerate(data_loader_test):
                images, targets = test_data
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
                accumulate_losses = {k: scaled_losses_dict[k] if k not in accumulate_losses else sum([accumulate_losses[k], scaled_losses_dict[k]]) for k, _ in scaled_losses_dict.items()}

                if i % print_logs_every == print_logs_every - 1:
                    accumulate_losses = {k: v.item() / print_logs_every for k, v in accumulate_losses.items()}
                    print(f'Test epoch [{epoch}] -> [{i}/{len(data_loader_test)}]: ' + ', '.join([f'{k}: {v}' for k, v in accumulate_losses.items()]))
                    temp_logs['test'].append(accumulate_losses)
                    accumulate_losses = {}

        epoch_total = {}
        for d in temp_logs['test']:
            for k in d:
                epoch_total[k] = epoch_total.get(k, 0) + d[k]

        logs['test'].append({k: v / len(temp_logs['test']) for k, v in epoch_total.items()})

        print(f'----> EPOCH SUMMARY TEST [{epoch}] -> [{i}/{len(data_loader_test)}]: ' + ', '.join([f'{k}: {v}' for k, v in logs['test'][epoch].items()]))

        #lr_scheduler.step()

    model.save(epoch=params['epochs'] - 1,
               optimizer_state_dict=optimizer.state_dict(),
               lr_scheduler_state_dict=lr_scheduler.state_dict(),
               params=params,
               logs=logs,
               )


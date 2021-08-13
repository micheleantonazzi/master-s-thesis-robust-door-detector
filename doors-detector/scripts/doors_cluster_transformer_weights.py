import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots, show
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detector.dataset.torch_dataset import DEEP_DOORS_2
from doors_detector.models.detr import PostProcess
from doors_detector.models.detr_door_detector import *
from doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import seed_everything, collate_fn
from scripts.dataset_configurator import get_deep_doors_2_labelled_sets

device = 'cuda'
seed_everything(0)
batch_size = 1
values = {'transformer': [], 'max_scores': [], 'labels': []}

model = DetrDoorDetector(model_name=DETR_RESNET50, pretrained=True, dataset_name=DEEP_DOORS_2, description=PRETRAINED_FINETUNE_ALL_LR_LOW_NOSTEP_AUG_10OBJQUERIES_LABELLED)
train, test, labels = get_deep_doors_2_labelled_sets()

print(model)

def extract_tranformer_weights(model, input, output):
    tensor = output[0].detach()
    b = torch.split(tensor[-1], 1, 0)
    values['transformer'].extend(b)

model.model.transformer.register_forward_hook(
    extract_tranformer_weights
)

model.to(device)
model.eval()

data_loader_training = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=6)

post_processor = PostProcess()

for i, training_data in tqdm(enumerate(data_loader_training), total=len(data_loader_training)):
    images, targets = training_data

    images = images.to(device)
    outputs = model(images)

    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([[100, 100] for _ in range(len(images))], device=device))
    for item in processed_data:
        scores = item['scores']
        max_score = torch.argmax(scores).item()
        values['max_scores'].append(max_score)
        values['labels'].append(item['labels'][max_score].item())

flatten_encoder = np.array([torch.squeeze(v)[m].flatten().tolist() for v, m in zip(values['transformer'], values['max_scores'])])
flatten_encoder_pca = PCA(n_components=50, random_state=42).fit_transform(flatten_encoder)

colors = np.array([
    "tab:red",
    "tab:blue",
    "tab:green"
])

fig, axes = subplots(nrows=2, ncols=3, figsize=(10, 5))
for perplexity, axis in tqdm(zip([30, 40, 50, 100, 500, 5000], axes.flatten()), desc="Computing TSNEs", total=6):
    axis.scatter(*TSNE(n_components=2, perplexity=perplexity).fit_transform(flatten_encoder_pca).T, s=1, color=colors[values['labels']])
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.set_title(f"TSNE decomposition - perplexity = {perplexity}", fontdict={'fontsize': 10,
                                                                                'fontweight': 10,
                                                                                'verticalalignment': 'baseline',
                                                                                'horizontalalignment': 'center'})
fig.tight_layout()
plt.show()

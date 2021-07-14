import torch


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def collate_fn(batch_data):
    # Batch data is a list of n tuple, where tuple[0] is the img while tuple[1] are targets (labels, bounding boxes ecc)
    # Batch data is transformed in a list where list[0] contains a list of the images and list[1] contains a list of targets
    batch_data = list(zip(*batch_data))

    # Replace batch_data[0] with a tensor containing all batch images
    batch_size = len(batch_data[0])
    image_size = list(batch_data[0][0].size())
    device = batch_data[0][0].device
    dtype = batch_data[0][0].dtype

    tensor = torch.zeros([batch_size, *image_size], dtype=dtype, device=device)

    for i, img in enumerate(batch_data[0]):
        tensor[i] = img.detach().clone()

    batch_data[0] = tensor

    return tuple(batch_data[:2])
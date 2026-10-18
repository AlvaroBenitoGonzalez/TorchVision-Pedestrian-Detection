import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from model import get_model_instance_segmentation
from transforms_utils import get_transform

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

model = get_model_instance_segmentation(num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

image = read_image("data/PennFudanPed/PNGImages/FudanPed00046.png")
eval_transform = get_transform(train=False)

with torch.no_grad():
    x = eval_transform(image)

    # convert RGBA to RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = {k: v.cpu() for k, v in predictions[0].items()}

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]

pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.savefig("result.png")
# TorchVision Object Detection: Pedestrian Instance Segmentation

Fine-tuning a **Mask R-CNN (ResNet-50 FPN)** model, pretrained on COCO, to detect and segment pedestrians using the [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/). The project follows the official [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), split into small, reusable modules.

For every pedestrian in an image, the model predicts:

- a **bounding box**
- a **segmentation mask**
- a **confidence score**

After running `setup.sh`, these files are also downloaded from the [TorchVision detection references](https://github.com/pytorch/vision/tree/main/references/detection): `engine.py`, `utils.py`, `coco_utils.py`, `coco_eval.py` and `transforms.py`.

## Requirements

- Python 3.10+
- PyTorch with `torch.accelerator` support (PyTorch 2.6 or newer)
- A GPU is recommended but not required. The scripts fall back to CPU automatically.
- `wget` and `unzip` (for `setup.sh`)

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# 4. Download the dataset and reference scripts
bash setup.sh
```

> `requirements.txt` pins CUDA 12.6 builds of PyTorch (`+cu126`), hence the extra index URL. If you want a CPU-only install or a different CUDA version, install `torch` and `torchvision` following the [official instructions](https://pytorch.org/get-started/locally/) and install the remaining packages manually.

## Usage

### Train

```bash
python train.py
```

What it does:

- Randomly splits the dataset: all images except the last 50 for training, 50 for testing.
- Fine-tunes Mask R-CNN with SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`) and a `StepLR` scheduler.
- Evaluates on the test split after every epoch (COCO-style metrics).
- Saves the weights to `model.pth`.

By default it trains for **2 epochs** with a batch size of 1. To get better results, increase `num_epochs` in `train.py`.

### Predict

```bash
python predict.py
```

This loads `model.pth`, runs the model on `data/PennFudanPed/PNGImages/FudanPed00046.png`, and writes the visualization (boxes, scores and masks) to `result.png`. To use your own image, change the path in `predict.py`.

Masks are binarized with a threshold of `0.7`. Since all predictions are drawn regardless of score, you may want to filter low-confidence detections (for example `pred["scores"] > 0.5`) for a cleaner output.

## How It Works

1. **Dataset** (`dataset.py`): each mask image encodes every pedestrian with a different pixel value. These are split into one binary mask per person, and bounding boxes are derived from the masks.
2. **Model** (`model.py`): loads `maskrcnn_resnet50_fpn` with default COCO weights, then replaces the box predictor (`FastRCNNPredictor`) and mask predictor (`MaskRCNNPredictor`) to output 2 classes: background and pedestrian.
3. **Transforms** (`transforms_utils.py`): random horizontal flip (training only) and conversion to float tensors scaled to `[0, 1]`.
4. **Training** (`train.py`): uses the `train_one_epoch` and `evaluate` helpers from the TorchVision references.

## Dataset

[Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/): 170 images with 345 labeled pedestrians. It is downloaded automatically into `data/` by `setup.sh` and is not included in this repository.

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [PyTorch / TorchVision](https://pytorch.org/vision/stable/index.html) and the [official detection finetuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html?utm_source=chatgpt.com#putting-everything-together)
- Penn-Fudan Database authors

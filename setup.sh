#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d data/PennFudanPed ]; then
    mkdir -p data
    wget -nc https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data
    unzip -n data/PennFudanPed.zip -d data
else
    echo "Dataset ya presente, se omite."
fi

base=https://raw.githubusercontent.com/pytorch/vision/main/references/detection
for f in engine.py utils.py coco_utils.py coco_eval.py transforms.py; do
    wget -nc "$base/$f"
done

echo "Listo. Ahora puedes ejecutar: python train.py"
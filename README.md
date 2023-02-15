# CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation

Re-implementation of the 3D version of the [original CS-Net](https://github.com/iMED-Lab/CS-Net) using [MONAI framework](https://github.com/Project-MONAI/MONAI). 

## Installation

```angular2html
conda create -n csnet python=3.9 \
conda activate csnet \
pip install monai[all] \
pip install git+https://github.com/amedyukhina/CSNet.git
```

## Usage

### Generate training data

```angular2html
python scripts/generate_training_data.py \
    --data-dir data --img-shape 32,64,64 \
    --subfolders train,val,test --n-img 400,100,10
```

For the full list of options, run:
```angular2html
python scripts/generate_training_data.py --help
```

### Training

```angular2html
python scripts/train.py \
    --data-dir data \
    --model-path model \
    --epochs 20 \
    --batch-size 2 \
    --lr 0.0001
```

For the full list of options, run:
```angular2html
python scripts/train.py --help
```

### Prediction

```angular2html
python scripts/predict.py \
    --input-dir data/test/img/ \
    --model-path model/MODEL_NAME/MODEL_CHECKPOINT.pth \
    --output-dir predictions/
```

For the full list of options, run:
```angular2html
python scripts/predict.py --help
```

Jupyter Notebook versions of these scripts are also [available](notebooks).


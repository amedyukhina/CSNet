# CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation

Re-implementation of the 3D version of the [original CS-Net](https://github.com/iMED-Lab/CS-Net). 

## Installation

```angular2html
conda create -n csnet python=3.9
conda activate csnet
pip install git+https://github.com/amedyukhina/CSNet.git
```

## Usage

### Generate training data

```angular2html
python scripts/generate_training_data.py \
    --data-dir data --img-shape 16,64,64 \
    --subfolders train,val,test --n-img 20,10,5
```

For the full list of options, run:
```angular2html
python scripts/generate_training_data.py --help
```

### Training

```angular2html
python scripts/train.py \
    --dir-train-gt data/train/gt/ \
    --dir-train-img data/train/img/ \
    --dir-val-gt data/val/gt/ \
    --dir-val-img data/val/img/ \
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
    --model-path model/MODEL_NAME/MODEL_CHECKPOINT.pkl \
    --output-dir predictions/
```

For the full list of options, run:
```angular2html
python scripts/predict.py --help
```

Jupyter Notebook version of these scripts are also [available](notebooks).


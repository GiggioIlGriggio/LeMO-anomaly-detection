# LeMO: a fully online image anomaly detection method

Unofficial PyTorch implementation of [Towards Total Online Unsupervised Anomaly Detection and Localization in Industrial Vision](https://arxiv.org/abs/2305.15652) (LeMO).
This work is part of a thesis in Artificial Intelligence.

## Getting Started

Install packages with:

```
$ pip install -r requirements.txt
```

## Dataset 

Prepare industrial image as:

``` 
train data:
    dataset_path/class_name/train/good/any_filename.png
    [...]

test data:
    dataset_path/class_name/test/good/any_filename.png
    [...]

    dataset_path/class_name/test/defect_type/any_filename.png
    [...]
``` 

## How to train

### Example
```
python trainer_lemo.py --class_name all --data_path [/path/to/dataset/] --cnn wrn50_2 --size 224 --gamma_c 1 --gamma_d 1 --loss NCENEW --memory_update kmeans
```

## Performance 
### WideResNet-50
R : resize. 
C : crop

|                |     Official LeMO     |      Ours      |
|----------------|-----------------------|----------------|
| Image AUROC    |           0.972       |      0.956     |
| Pixel AUROC    |           0.976       |      0.970     |

For more details about the model performances see the LeMO_report.pdf, chapter 5.2.5.


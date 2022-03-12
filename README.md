# Lipreading using Temporal Convolutional Networks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-practical-lipreading-with-distilled/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=towards-practical-lipreading-with-distilled)

## Content
[Introduction](#introduction)

[Preprocessing](#preprocessing)

[How to install the environment](#how-to-install-environment)

[How to prepare the dataset](#how-to-prepare-dataset)

[How to train](#how-to-train)

[How to test](#how-to-test)

[Citation](#citation)

[License](#license)



## IDL FINAL PROJECT TEAM4

### Introduction

```Shell
$LIPS directory structure

ㄴconfigs
	follows the structure of lrw_resnet18_mstcn.yaml

ㄴdata
	this is where we keep all our datasets including…
	lipread_mp4 (LRW dataset)
	visual_data (preprocessed LRW landmarks)

ㄴdatasets
	this is where we keep our dataloaders that inherit base_dataset.py
	lrw_dataset is the dataloader for LRW dataset

ㄴlabels
	labels for dataloader

ㄴlandmarks
	original <RW landmarks

ㄴmodels
	this is where we keep our models that inherit base_model.py
	Each model is indexed by its name ‘lipreading’ and 
	is a class with the following functions:
	__init__, forward, learn, evaluate 
	lipreading.py is the default model that we use
	
ㄴpreprocessing
	data augmentation

ㄴutils
	dataloading and training utils

ㄴdata.py
	datascheduler used in train/test

ㄴmain
	includes train_model, test_model in one main file
ㄴtrain.py
ㄴtest.py
ㄴrequirements.txt
```

### Preprocessing

As described in [the paper](https://arxiv.org/abs/2001.08702), each video sequence from the LRW dataset is processed by 1) doing face detection and face alignment, 2) aligning each frame to a reference mean face shape 3) cropping a fixed 96 × 96 pixels wide ROI from the aligned face image so that the mouth region is always roughly centered on the image crop 4) transform the cropped image to gray level.

You can run the pre-processing script provided in the [preprocessing](./preprocessing) folder to extract the mouth ROIs.

### How to install environment

1. Clone the repository into a directory. We refer to that directory as *`$LIPS`*.

```Shell
git clone https://github.com/ib33x99/lips.git
```

2. Install all required packages.

```Shell
pip install -r requirements.txt
```

### How to prepare dataset

1. Download the pre-computed landmarks from [GoogleDrive](https://bit.ly/3huI1P5) or [BaiduDrive](https://bit.ly/2YIg8um) (key: kumy) and unzip them to *`$LIPS/landmarks/`* folder.

2. Pre-process mouth ROIs using the script [crop_mouth_from_video.py](./preprocessing/crop_mouth_from_video.py) in the [preprocessing](./preprocessing) folder and save them to *`$LIPS/data/visual_data/`*.

3. Locate the original LRW dataset that includes timestamps (.txt) to *`$LIPS/data/lipread_mp4`*.

4. Before running the code, make *`$LIPS/logs`* directory and link subdirectory of logs with *`$LIPS/log`*. See the following shell for an example code.

```Shell
mkdir logs
mkdir logs/1225
ln -s logs/1225 log
```
(rm log erases the symlink without deleting the original logs directory)

### How to train

1. Train a visual-only model. This saves the config file you used inside the specified log directory.

```Shell
python main --config configs/lrw_resnet18_mstcn.yaml -l log/train_visual
```

2. To exclude the need of editing the config file every time, you can use *`--override`*.

```Shell
python main --config configs/lrw_resnet18_mstcn.yaml --override "num_workers=0" -l debug/1

python main --config configs/lrw_resnet18_mstcn.yaml --override "eval_step=1|summary_step=100|ckpt_step=1000" -l debug/2
```

3. Resume from checkpoint.

You can pass the checkpoint path (.ckpt) *`<CHECKPOINT-PATH>`* to the variable argument *`--resume-ckpt`* or *`--resume-latest-ckpt`*, and specify the ckpt path or log directory to resume training.


### How to test

Evaluate the visual-only performance (lipreading).

```Shell
python main --config configs/lrw_resnet18_mstcn.yaml --resume-latest-ckpt log/train_visual -l log/test_visual --test
```

## Citation

Most of the code is based on the following papers:

```bibtex
@INPROCEEDINGS{ma2020towards,
  author={Ma, Pingchuan and Martinez, Brais and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Towards Practical Lipreading with Distilled and Efficient Models},
  year={2021},
  pages={7608-7612},
  doi={10.1109/ICASSP39728.2021.9415063}
}

@INPROCEEDINGS{martinez2020lipreading,
  author={Martinez, Brais and Ma, Pingchuan and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Lipreading Using Temporal Convolutional Networks},
  year={2020},
  pages={6319-6323},
  doi={10.1109/ICASSP40776.2020.9053841}
}
```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

# About This Project

## Overview

The Age Estimator and Face matching project utilizes computer vision techniques to estimate the age of individuals based on facial images and apply face matchine for the input images. This repository contains the code to preprocess a dataset of images, and load the data for further analysis or model training.

## Download Code
clone the repository:
```bash
# git clone https://github.com/Bassantraafat98/Agepredection_withFaceMatching.git
```

## Requirements
- install requirements.txt

## Age Estimation
- **Used Dataset**:The used dataset is UTK dataset, which consists of images categorized by age, gender, and ethnicity. Please download from this link and use `utkcropped` folder to create csv: [Kaggle](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped?select=utkcropped)

- **Preparing the dataset**: 
Automatically generates a CSV file containing image names, ages, ethnicities, and genders extracted from image filenames.

1-create a csv file which contains labels using next command
```bash
python Age_Estimator/Age_create_csv.py
```

- **Model Traning**: for model Traning you need to setup the configuration parameters in Age_config.py, this repo contains 2 models for Age estimation Resnet50, and Various Auto Encoder
for model traning and evaluation
```bash
python Age_Estimator/Age_Training.py
```

## Face Matching
Face matching using deep learning (CNN embedding + triplet loss)

- **Used Dataset**:The used dataset is Labeled Faces in the Wild (LFW) dataset, Labeled Faces in the Wild (LFW) is a database of face photographs designed for studying the problem of unconstrained face recognition.
For dataset downloading follow this link https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

- **Preparing the dataset**: 

split your data into train and test and put it this way under the same project folder.,
run this command
```bash
python Face_matching/split_data.py
```
```bash
/lfw-deepfunneled_splitted
    ├── train
    └── test
```
- **Model Traning**: for model Traning you need to setup the configuration parameters in Face_config.py.

for model traning and evaluation

```bash
python Face_matching/FaceMatch_main.py 
```

## Test

After training, you can test the models on a pair of images (to decide there age, and whether they belong to the same person or no).but before decompress the folder checkpoints in Age_Estimator checkpoints_AgeDetect.zip, and in Face_matching checkpoints_Facematch.zip

```bash
python Predection_FaceAge.py --path_img1  imgPath1.png --path_img2 imgPath2.png
```

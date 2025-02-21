#!/usr/bin/env python3
import torchvision
import torch
import PIL
import numpy as np
from packaging import version
import platform
from pathlib import Path
from Face_matching.FaceMatch import FaceEmb
import Age_Estimator.prediction_Age as Age_pred 
import glob
import os
from Age_Estimator.Age_Detector_model import AgeEstimationModel
from Age_Estimator.Age_config import config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img1', type=Path)
    parser.add_argument('--path_img2', type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 1.1 # thresold to classify the imag to positive or negative (set to 1.1 same to facenet)
    transforms = torchvision.transforms.ToTensor()

    # Read the two images and transform them to tensors
    img1 = PIL.Image.open(args.path_img1).convert('RGB')
    img1 = img1.resize((250, 250)) 
    img1 = transforms(img1)
    img1 = img1.unsqueeze(0)
    img1 = img1.to(device)
    
    img2 = PIL.Image.open(args.path_img2).convert('RGB')
    img2 = img2.resize((250, 250)) 
    img2 = transforms(img2)
    img2 = img2.unsqueeze(0)
    img2 = img2.to(device)

    # define your model
    model = FaceEmb()
    model = model.to(device)
    model.load_state_dict(torch.load('./Face_matching/checkpoints_Facematch/model.pt'))

    # embedd the images into vectors
    out_img1 = model.backbone(img1)
    out_img2 = model.backbone(img2)
    
    # compute euclidian distance
    distance = torch.cdist(out_img1, out_img2)
    distance = distance.item()

    # decide whether it is the same person or it is different
    if distance <= threshold:
        print("Same, distance = ", distance)
    else:
        print("Different, distance =", distance)


    path = "./Age_Estimator/checkpoints_AgeDetect"
    checkpoint_files = glob.glob(os.path.join(path, 'epoch-*-loss_valid-*.pt'))
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    pretrain_weights='IMAGENET1K_V2'
    # pretrain_weights=False
    model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights=pretrain_weights).to(config['device'])
    # model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='vit', pretrain_weights=pretrain_weights).to(config['device'])

    # Load the model using the latest checkpoint file
    model.load_state_dict(torch.load(latest_checkpoint))

    image_path_test_1 = args.path_img1 # Path to the input image
    output_path_test1_output= str(image_path_test_1).split('.')[0]+"_output1.png"               # Path to save the output image
    image_path_test_2  = args.path_img2  # Path to save the output image
    output_path_test2_output=str(image_path_test_2).split('.')[0]+"_output2.png" 
    Age_pred.inference(model, image_path_test_1, output_path_test1_output)
    Age_pred.inference(model, image_path_test_2, output_path_test2_output)
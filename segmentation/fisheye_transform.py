import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from PIL import Image

import cv2

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser

parser = ArgumentParser(description="fisheye")
parser.add_argument('--seed', default=826, type=int)
args = parser.parse_args('')
SEED = args.seed

def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)

set_seeds()

train_df = pd.read_csv('data/train_source.csv')
val_df = pd.read_csv('data/val_source.csv')
test_df = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

train_df["img_path"] = train_df["img_path"].apply(lambda x : "data/" + x[2:])
train_df["gt_path"] = train_df["gt_path"].apply(lambda x : "data/" + x[2:])
val_df["img_path"] = val_df["img_path"].apply(lambda x : "data/" + x[2:])
val_df["gt_path"] = val_df["gt_path"].apply(lambda x : "data/" + x[2:])
test_df["img_path"] = test_df["img_path"].apply(lambda x : "data/" + x[2:])

## Mean Target Image

print("Mean Target Image Generation")

train_target_image_path = 'data/train_target_image/'

target_images = []
for i in tqdm(os.listdir(train_target_image_path)):
    image = Image.open(train_target_image_path+i).convert('RGB')
    target_images.append(np.array(image))

mean_target_image = np.mean(target_images, axis=0).round().astype("uint8")

## Mask Target GT

print("Mask Target GT Generation")

mean_target_gt = np.mean(mean_target_image, axis=2).round().astype("uint8")
mask_target_gt = np.array(mean_target_gt>108, bool)
mask_target_gt[360:-360, 120:-120] = True

## Fisheye Transformation

print("Fisheye Transformation: train_source_image")

train_source_image_path = 'data/train_source_image/'

for i in tqdm(os.listdir(train_source_image_path)):
    image = Image.open(train_source_image_path+i).convert('RGB').resize((1920, 1080), Image.NEAREST) 
    image = np.array(image)
    image1 = image * mask_target_gt[:, :, np.newaxis]
    image2 = mean_target_image * np.where(mask_target_gt, False, True)[:, :, np.newaxis]
    image = image1 + image2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("data/preprocess_data/train_fisheye_image/"+i, image)

print("Fisheye Transformation: val_source_image")

val_source_image_path = 'data/val_source_image/'

for i in tqdm(os.listdir(val_source_image_path)):
    image = Image.open(val_source_image_path+i).convert('RGB').resize((1920, 1080), Image.NEAREST) 
    image = np.array(image)
    image1 = image * mask_target_gt[:, :, np.newaxis]
    image2 = mean_target_image * np.where(mask_target_gt, False, True)[:, :, np.newaxis]
    image = image1 + image2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("data/preprocess_data/val_fisheye_image/"+i, image)

print("Fisheye Transformation: train_source_gt")

train_source_gt_path = 'data/train_source_gt/'

for i in tqdm(os.listdir(train_source_gt_path)):
    image = Image.open(train_source_gt_path+i).convert('L').resize((1920, 1080), Image.NEAREST) 
    image = np.array(image)
    image+=1
    image = image * mask_target_gt
    cv2.imwrite("data/preprocess_data/train_fisheye_gt/"+i, image)

print("Fisheye Transformation: val_source_gt")

val_source_gt_path = 'data/val_source_gt/'

for i in tqdm(os.listdir(val_source_gt_path)):
    image = Image.open(val_source_gt_path+i).convert('L').resize((1920, 1080), Image.NEAREST) 
    image = np.array(image)
    image+=1
    image = image * mask_target_gt
    cv2.imwrite("data/preprocess_data/val_fisheye_gt/"+i, image)

print("end")
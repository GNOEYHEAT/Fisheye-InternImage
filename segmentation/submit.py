import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from PIL import Image

from tqdm import tqdm

import pickle

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser

parser = ArgumentParser(description="Submission")
parser.add_argument('--pkl_name', default="exp_xx", type=str)
args = parser.parse_args()

pkl_name = args.pkl_name + ".pkl"

print("pickle load start")

# load
with open(f'results/{pkl_name}', 'rb') as f:
    y_preds = pickle.load(f)

y_preds = np.stack(y_preds)

print("rle encoding start")

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

result = []
for pred in tqdm(y_preds):
    pred = pred.astype(np.uint8)
    pred -= 1 # 0, background encode
    pred = Image.fromarray(pred)
    pred = pred.resize((960, 540), Image.NEAREST)
    pred = np.array(pred)
    for class_id in range(12):
        class_mask = (pred == class_id).astype(np.uint8)
        if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
            mask_rle = rle_encode(class_mask)
            result.append(mask_rle)
        else: # 마스크가 존재하지 않는 경우 -1
            result.append(-1)

if not os.path.exists("submission/"):
    os.makedirs("submission/")

sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission['mask_rle'] = result
sample_submission.to_csv(f'submission/{args.pkl_name}.csv', index=False)

print("submission end")
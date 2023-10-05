# Fisheye-InternImage

The following codes are the solutions **(3st place, private score: 0.66886)** for the dacon competition.

If you would like to know more about the competition, please refer to the following link:

[2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation](https://dacon.io/competitions/official/236132/overview/description)

* 주최 : Samsung Advanced Institute of Technology
* 주관 : DACON

## Usage
Change to the **"segmentation"** directory and follow the **README.md**

## Directory Structure
<pre><code>
/workspace
├── configs
│   ├── _base_
│   │   ├── datasets
│   │   │   ├── samsung_fisheye.py
│   ├── samsung
│   │   ├── exp_01.py
│   │   ├── exp_02.py
│   │   ├── exp_03.py
│   │   ├── exp_04.py
│   │   ├── exp_05.py
├── data
│   ├── preprocess_data
│   │   ├── train_fisheye_gt
│   │   ├── train_fisheye_image
│   │   ├── val_fisheye_gt
│   │   ├── val_fisheye_image
│   ├── test_image
│   ├── train_source_gt
│   ├── train_source_image
│   ├── train_target_image
│   ├── val_source_gt
│   ├── val_source_image
│   ├── sample_submission.csv
│   ├── test.csv
├── results
│   ├── exp_04.pkl
│   ├── exp_05.pkl
├── submission
│   ├── exp_04.csv
│   ├── exp_05.csv
├── fisheye_transform.py
├── submit.py
├── test.py
├── train.py
      .
      .
      .
</code></pre>

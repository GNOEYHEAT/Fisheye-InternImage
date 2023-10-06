# Fisheye-InternImage

The following codes are the solutions **(3st place, private score: 0.66886)** for the dacon competition.

If you would like to know more about the competition, please refer to the following link:

[2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation](https://dacon.io/competitions/official/236132/overview/description)

* 주최 : Samsung Advanced Institute of Technology
* 주관 : DACON

## Usage
Move to the **"segmentation"** directory and follow the *README.md*

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
│   ├── train_source.csv
│   ├── train_target.csv
│   ├── val_source.csv
├── results
│   ├── exp_04.pkl
│   ├── exp_05.pkl
├── submission
│   ├── exp_04.csv
│   ├── exp_05.csv
├── work_dirs
│   ├── exp_04
│   │   ├── best_mIoU_iter_22000.pth
├── fisheye_transform.py
├── submit.py
├── test.py
├── train.py
      .
      .
      .
</code></pre>

## Experiments

The final submission is **exp_04**.

| Index      | Model       | Private mIoU | Public mIoU | Val mIoU (%) | Iter  | Fine-tuned model | Pre-trained model |
|------------|-------------|--------------|-------------|--------------|-------|------------------|-------------------|
| exp_01     | UperNet     | 0.64483      | 0.6192      | 67.30        | 20000 | [ckpt]()         | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_jointto22k_384.pth) |
| exp_02     | Mask2Former | 0.66456      | 0.62771     | 66.01        | 9000  | [ckpt]()         | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff164k.pth) |
| exp_03     | UperNet     | 0.65114      | 0.62775     | 68.03        | 14000 | [ckpt]()         | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_jointto22k_384.pth) |
| **exp_04** | Mask2Former | **0.66886**  | 0.63133     | 70.34        | 22000 | [ckpt]()         | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff164k.pth) |
| exp_05     | Mask2Former | 0.67288      | 0.62905     | -            | 40000 | [ckpt]()         | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff164k.pth) |

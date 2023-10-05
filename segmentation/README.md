# InternImage for Semantic Segmentation

This folder contains the implementation of the InternImage for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/GNOEYHEAT/Fisheye-InternImage.git
cd Fisheye-InternImage/segmentation
```

- Create a conda virtual environment and activate it:

```bash
conda create -n fisheye python=3.7 -y
conda activate fisheye
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3 and nvcc:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev=11.3 -y # to install nvcc
```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip. 	  

```bash
conda install -c conda-forge termcolor yacs pyyaml scipy seaborn pip -y 
pip install opencv-python
pip install yapf==0.40.1

```

- Install `timm` and `mmcv-full` and `mmsegmentation':

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1

if "ERROR: Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall." error occurs when installing openmim:

conda remove pyyaml
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Compile CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
- You can also install the operator using .whl files
[DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

### Data Preparation

Prepare datasets according to the [guidelines] (https://github.com/GNOEYHEAT/Fisheye-InternImage/blob/main/README.md)

### FishEye Transformation
To transform original data into fisheye-data, run:
```bash
python fisheye_transform.py
```

### Inference

To evaluate our model on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```
You can download checkpoint files from [here](https://huggingface.co/OpenGVLab/InternImage/tree/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91). Then place it to segmentation/checkpoint_dir/seg.

For example, to evaluate the `InternImage-T` with a single GPU:

```bash
python test.py configs/ade20k/upernet_internimage_t_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth --eval mIoU
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/samsung/upernet_internimage_b_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_b_512_160k_ade20k.pth 8 --eval mIoU
```

### Training

To train our model on preprocess_data, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `이름 수정(exp_01)` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/samsung/exp_01
```
### Citation
```javascript
<font color="red">@article</font>{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}
```


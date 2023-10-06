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

Prepare datasets according to the [directory structure guideline](https://github.com/GNOEYHEAT/Fisheye-InternImage/blob/main/README.md)

### Fisheye Transformation
To transform original data into fisheye-data, run:
```bash
python fisheye_transform.py
```

### Submission Guideline

To submit our model on test data, run:

```bash
python test.py <config-file> <checkpoint> --out <pickle-file>
python submit.py --pkl_name <pickle-file>
```

You can download checkpoint files from [here](https://huggingface.co/lamar041523/Fisheye-InternImage/resolve/main/exp_04/best_mIoU_iter_22000.pth). Then place it to segmentation/work_dirs/exp_04.

For example, to inference the `exp_04`:

```bash
python test.py configs/samsung/exp_04.py work_dirs/exp_04/best_mIoU_iter_22000.pth --out results/exp_04.pkl
```

For example, to submit the `exp_04`:

```bash
python submit.py --pkl_name exp_04
```

### Training

To train our model on preprocess_data, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `exp_04` with 2 GPU on 1 node (total batch size 4), run:

```bash
sh dist_train.sh configs/samsung/exp_04.py 2 -seed 826
```
### Citation
```bibtex
@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}
```
### Acknowledgements
This code is heavily borrowed from [**INTERNIMAGE**](https://github.com/OpenGVLab/InternImage)

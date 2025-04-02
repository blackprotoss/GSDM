## Text Image Inpainting via Global Structure-Guided Diffusion Models (Accepted by AAAI-24)

*[Shipeng Zhu](http://palm.seu.edu.cn/homepage/zhushipeng/demo/index.html), [Pengfei Fang](https://fpfcjdsg.github.io/), Chenjie Zhu, [Zuoyan Zhao](http://palm.seu.edu.cn/homepage/zhaozuoyan/index.html), Qiang Xu, [Hui Xue](http://palm.seu.edu.cn/hxue/)*

Paper: [(arXiv 2401.14832)](https://arxiv.org/abs/2401.14832), [(AAAI-24)](https://ojs.aaai.org/index.php/AAAI/article/view/28612)

This repository offers the official Pytorch code for this paper. If you have any questions, feel free to contact Shipeng Zhu (shipengzhu@seu.edu.cn) or Chenjie Zhu (chenjiezhu@seu.edu.cn).


## Environment Setup

![python](https://img.shields.io/badge/Python-v3.10-green.svg?style=plastic)  ![pytorch](https://img.shields.io/badge/Pytorch-v1.13.1-green.svg?style=plastic)  ![cuda](https://img.shields.io/badge/Cuda-v11.6-green.svg?style=plastic)

* Clone this repo

* Create a conda environment and activate it.

* Install related version Pytorch following

  ```
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
  ```

* Install the required packages

* Download the pre-trained checkpoints, and and move these files into the "checkpoints".

## Inference phase

```python
python inference.py --config xx --input_dir input --output_dir output --save_sp False
```

* config: The path loading yaml file.
* input_dir: The input image path.
* output_dir: The Output image path.
* save_sp: Whether to save structure prediction images.

## Datasets and Pre-trained Checkpoints

- Download the TII-HT and TII-ST datasets from: [Baidu Cloud](https://pan.baidu.com/s/1ENLY0pn3amnlOvi4GzNdzg), Passwd: h5i0; [Google Drive](https://drive.google.com/drive/folders/1eN8Mn1wruhhu98wWJmaQOmYvQfbR0Z20?usp=sharing).
- Download the Checkpoints from: [Baidu Cloud](https://pan.baidu.com/s/1MiyY50A2dGy0wyndYonHUA ), Passwd: dlr6; [Google Drive](https://drive.google.com/drive/folders/1ykYNzv-aYltC5I36T6SqvGWwg8no6uhY?usp=sharing).

## Training phase

#### Step 1: Training SPM

```python
python train_spm.py
```

* Modify the training configuration in this file ——"config/train_spm.yaml"

#### Step 2: Training RM

```
python train_rm.py
```

* Modify the training configuration in this file ——"config/train_rm.yaml"
* Note that training RM requires a pre-trained SPM checkpoint,  and the path should be modified in the above file.
* Download the checkpoint of the pre-trained CRNN model into the path: "crnn/data/"


## Citation

  ```
@inproceedings{zhu2024gsdm,
  title={Text image inpainting via global structure-guided diffusion models},
  author={Zhu, Shipeng and Fang, Pengfei and Zhu, Chenjie and Zhao, Zuoyan and Xu, Qiang and Xue, Hui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7775-7783},
  year={2024}
}
  ```

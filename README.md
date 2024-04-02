


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

* Download the pre-trained checkpoints

## Inference phase

* config: The path loading yaml file.
* ckpt_dir: The model checkpoints saving directory.
* input_dir: The input image path.
* output_dir: The Output image path.
* save_sp: Whether to save structure prediction images.

## Datasets and Pre-trained Checkpoints

- Download the TII-HT and TII-ST datasets from: [Baidu Cloud](https://pan.baidu.com/s/1ENLY0pn3amnlOvi4GzNdzg), Passwd: h5i0 
- Download the Checkpoints from: [Baidu Cloud](https://pan.baidu.com/s/1MiyY50A2dGy0wyndYonHUA ), Passwd: dlr6; [Google Drive](https://drive.google.com/drive/folders/1ykYNzv-aYltC5I36T6SqvGWwg8no6uhY?usp=sharing).

## Todo List
- [X] Datasets
- [X] Inference Code
- [X] Pre-trained Checkpoints
- [ ] Training Code

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


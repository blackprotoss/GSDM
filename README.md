


## Text Image Inpainting via Global Structure-Guided Diffusion Models (Accepted by AAAI-24)

*[Shipeng Zhu](http://palm.seu.edu.cn/homepage/zhushipeng/demo/index.html), [Pengfei Fang](https://fpfcjdsg.github.io/), Chenjie Zhu, [Zuoyan Zhao](http://palm.seu.edu.cn/homepage/zhaozuoyan/index.html), Qiang Xu, [Hui Xue](http://palm.seu.edu.cn/hxue/)*


### Prerequisites 

* Python 3.10
* Pytorch 1.13.1
* CUDA 11.6

### Environment Setup

* Clone this repo

* Create a conda environment and activate it.

* Install related version Pytorch following

  ```
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
  ```

* Install the required packages

* Download the pre-trained checkpoints

### Inference phase

* config: The path loading yaml file.
* ckpt_dir: The model checkpoints saving directory.
* input_dir: The input image path.
* output_dir: The Output image path.
* save_sp: Whether to save structure prediction images.


## Datasets and Pre-trained Recognizers

- Download the TII-HT and TII-ST datasets from: [Baidu Cloud](https://pan.baidu.com/s/1ENLY0pn3amnlOvi4GzNdzg), Passwd: h5i0.



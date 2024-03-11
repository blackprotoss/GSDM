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

* Download the pretrained checkpoints

### Inference phase

* config : The path loading yaml file.
* ckpt_dir : The model checkpoints saving directory.
* input_dir : The input image path.
* output_dir : The Output image path.
* save_sp : Whether to save structure prediction images.
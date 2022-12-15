# gravitational_wave_data_2D_analysis

Gravitational wave data analysis and signal detection (in time-frequence images) with 2D AI model

## Introduction

This research proposed a new vision on Time series classification (TSC) tasks of low SNR (-20dB)  multi-source gravitational wave (GW) simulation data, which converts  1-dimension time series data into 2-dimension time-frequency image and then extracts the latent features. We mainly concerned on four different wave sources which are  extreme-mass-ratio-inspirals (EMRI), massive black hole binaries (MBHB), binary white dwarfs (BWD), and stochastic gravitational wave background (SGWB), the time-frequency representation of each source is shown as follows. 

**SGWB**

<img src="/Images/sgwb.JPEG" width="300" height="200"/><br/>

**MBHB**

<img src="/Images/smbhb.JPEG" width="300" height="200"/><br/>

**BWD**

<img src="/Images/bwd.JPEG" width="300" height="200"/><br/>

**EMRI**

<img src="/Images/emri.JPEG" width="300" height="200"/><br/>

**Noise**

<img src="/Images/noise.JPEG" width="300" height="200"/><br/>

## Installation

This repository is implemented based on the original Swin Transformer. The environment configuration  can refer to [the Swin's instruction](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md)

```
pip install install pytorch==1.8.0 torchvision==0.9.0 timm==0.4.12
```

Install other requirements

```
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

## Example Usage

### Data generation

The dataset is transfered from the generated 1-dimension GW simulation data, the data_generation.py can be directly ran to obtain the training data.

### Training on time-frequency  dataset

```
python -m torch.distributed.launch  --nproc_per_node 6  Swin-transformer/main.py --cfg Swin-transformer/configs/swin_base_patch4_window7_224.yaml  --data-path "your-path"   --cache-mode part  --batch-size 128
```

## Evaluation

Without extra steps like denoising, our model achieves over 90% accuracy on validation set, with a classification accuracy close to 100% for EMRI and BWD.

<img src="/Images/confusion_matrix.png" width="300" height="200"/><br/>

As can be observed, the SGWB is similar to noise regarding the data characteristics therefore results in a higher misidentification error. Since  the amplitude of noise is much higher than that of SGWB across the whole time series, the time-frequency representations of these types are extremely similar in waveform.

<img src="/Images/signal_compare.jpg" width="300" height="200"/><br/>

## Project Support

This material is based upon work supported by PengCheng Lab.


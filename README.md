##  A Data-Centric Solution to NonHomogeneous Dehazing via Vision Transformer

This is the official PyTorch implementation of A Data-Centric Solution to NonHomogeneous Dehazing via Vision Transformer.  
See more details in  [[report]](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Ancuti_NTIRE_2023_HR_NonHomogeneous_Dehazing_Challenge_Report_CVPRW_2023_paper.pdf) , [[paper]](https://arxiv.org/pdf/2304.07874.pdf), [[certificate]](https://cvlai.net/ntire/2023/NTIRE2023awards_certificates.pdf)

Our solution competes in NTIRE 2023 HR Non-homogeneous Dehazing Challenge, achieving the BEST performance in terms of PNSR, SSIM and LPIPS.

## News
- **(2023/6/18)**  We are the winner of the [NTIRE 2023 HR Non-homogeneous Dehazing Challenge!](https://cvlai.net/ntire/2023/)
- **(2022/6/01)**  We are invited to present our method at the [NTIRE 2023 HR Non-homogeneous Dehazing Challenge.](https://cvlai.net/ntire/2023/)

  
### Dependencies and Installation

* python3.7
* PyTorch >= 1.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* tensorboardX(optional)

### Pretrained Weights & Dataset

- Download [ImageNet pretrained SwinTransformer V2 weights](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth) and [our model weights](https://drive.google.com/file/d/1Nx5RpWA6CLqqLpsTrXCvEVcgh9l899ts/view?usp=share_link). 
- Download our [dataset](https://drive.google.com/drive/folders/1NwWRuQ8kWeCCkRsMv0IANFpDD40I_kyR?usp=share_link) or the original [dataset & gamma correction code](https://drive.google.com/drive/folders/1spwHOudYRfrMb2x4nKK2r5VRcgDx1-zD?usp=drive_link)


Please put train/val/test three folders into a root folder. This root folder would be the training dataset. 
Note, we have performed preprocessing to the data in folder train.

NTIRE2023_Val, and NTIRE2023_Test contain official validation and test. If you want to obtain val and test accuracy, please step towards the official [competition server.](https://codalab.lisn.upsaclay.fr/competitions/10216)

  
#### Train
```shell
python train.py --data_dir data --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml -train_batch_size 8 --model_save_dir train_result -train_epoch 6500
```

#### Test
 ```shell
python test.py --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml --model_save_dir ./output_img/test/best_result --ckpt_path ./checkpoints/best.pkl --hazy_data NTIRE2023_Test --cropping 4
 ```

* Using this command line for generating outputs of test data, the dehazed results could be found in: ./output_img/test/best_result
* This testing command line requires GPU memory >= 40 GB to ensure best results
  If GPU memory < 40 GB, please use " --cropping 6 " instead


## Qualitative Results

Results on NTIRE 2023 NonHomogeneous Dehazing Challenge test data:

<div style="text-align: center">
<img alt="" src="/images/test_results.PNG" style="display: inline-block;" />
</div>

## Citation
If you use the code in this repo for your work, please cite the following bib entries:

```latex
@inproceedings{liu2023data,
  title={A Data-Centric Solution to NonHomogeneous Dehazing via Vision Transformer},
  author={Liu, Yangyi and Liu, Huan and Li, Liangyan and Wu, Zijun and Chen, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1406--1415},
  year={2023}
}
```




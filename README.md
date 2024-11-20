# SSCFormer

This repository contains the supported pytorch code and configuration files to reproduce of SSCFormer.

![SSCFormer](img/Architecture_overview.jpg?raw=true)

Parts of codes are borrowed from [nn-UNet](https://github.com/MIC-DKFZ/nnUNet). For detailed configuration of the dataset, please refer to [nn-UNet](https://github.com/MIC-DKFZ/nnUNet).

## Environment

Please prepare an environment with Python 3.7, Pytorch 1.7.1, and Windows 10.

## Dataset Preparation

Datasets can be acquired via following links:

- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- [Brain_tumor](http://medicaldecathlon.com/)
- [Heart](http://medicaldecathlon.com/)

## Dataset Set

After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./SSCFormer/
./DATASET/
  ├── SSCFormer_raw/
      ├── SSCFormer_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task04_Heart/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── SSCFormer_cropped_data/
  ├── SSCFormer_trained_models/
  ├── SSCFormer_preprocessed/
```

## Preprocess Data

- SSCFormer_convert_decathlon_task -i D:\Codes\Medical_image\UploadGitHub\SSCFormer\DATASET\SSCFormer_raw\SSCFormer_raw_data
- SSCFormer_plan_and_preprocess -t 2

## Functions of scripts

- **Network architecture:**
  - SSCFormer\SSCFormer\network_architecture\SSCFormer_acdc.py``
  - SSCFormer\SSCFormer\network_architecture\SSCFormer_synapse.py``
  - SSCFormer\SSCFormer\network_architecture\SSCFormer_tumor.py``
  - SSCFormer\SSCFormer\network_architecture\SSCFormer_heart.py``
- **Trainer for dataset:**
  - SSCFormer\SSCFormer\training\network_training\SSCFormerTrainerV2_SSCFormer_acdc.py``
  - SSCFormer\SSCFormer\training\network_training\SSCFormerTrainerV2_SSCFormer_synapse.py``
  - SSCFormer\SSCFormer\training\network_training\SSCFormerTrainerV2_SSCFormer_tumor.py``
  - SSCFormer\SSCFormer\training\network_training\SSCFormerTrainerV2_SSCFormer_heart.py``

## Train Model

- python run_training.py  3d_fullres  SSCFormerTrainerV2_SSCFormer_synapse 2 0


## Test Model

- python predict.py -i D:\Codes\Medical_image\UploadGitHub\SSCFormer\DATASET\SSCFormer_raw\SSCFormer_raw_data\Task002_Synapse\imagesTs
  -o D:\Codes\Medical_image\UploadGitHub\SSCFormer\DATASET\SSCFormer_raw\SSCFormer_raw_data\Task002_Synapse\imagesTs_infer
  -m D:\Codes\Medical_image\UploadGitHub\SSCFormer\DATASET\SSCFormer_trained_models\SSCFormer\3d_fullres\Task002_Synapse\SSCFormerTrainerV2_SSCFormer_synapse__SSCFormerPlansv2.1
  -f 0

- python SSCFormer/inference_synapse.py

## Acknowledgements

This repository makes liberal use of code from:

- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) 
- [nnFormer](https://github.com/282857341/nnFormer)
- [UNETR++](https://github.com/Amshaker/unetr_plus_plus)

## Citation

@article{xie2024sscformer,  
  title={SSCFormer: Revisiting ConvNet-Transformer Hybrid Framework from Scale-Wise and Spatial-Channel-Aware Perspectives for Volumetric Medical Image Segmentation},  
  author={Xie, Qinlan and Chen, Yong and Liu, Shenglin and Lu, Xuesong},  
  journal={IEEE Journal of Biomedical and Health Informatics},  
  year={2024},  
  publisher={IEEE}  
}

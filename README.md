# S3-OCTA

中文版README: [README_zh](./README_zh.md)

An end-to-end deep learning __PyTorch__ framework for retinal vessel segmentation in optical coherence tomography angiography (OCTA) images.

## 1. Brief Introduction

### 1.1 Key Features

- __Dynamic Snake Convolution (DSConv)__ backbone specifically designed for tubular vessel structures.
- __Centerline extraction__ using boundary peeling based on Euler characteristics.
- __Custom loss function__ leveraging extracted centerlines for enhanced training.

### 1.2 Datasets and Related Papers

- __OCTA-500__: https://www.sciencedirect.com/science/article/abs/pii/S1361841524000173
- __ROSE__: https://ieeexplore.ieee.org/abstract/document/9284503
- __OCTA-25K__: https://arxiv.org/abs/2107.10476

### 1.3 GPU Memory

Different configurations consume different GPU memory. You can modify model depth and channel count in train.py:

- **Recommended**: 12 GB GPU.
- **Optimal**: 20 GB GPU for best performance.
- ⚠️ **Note**: Performance significantly degrades with less than 3GB GPU memory


## 2. Model Training

### Dataset Organization:
__OCTA-500__:

    /assets
        /datasets
            /OCTA-500
                /Artery
                    10101.bmp
                    10102.bmp
                    ...
                /RV
                    10101.bmp
                    10102.bmp
                    ...
                /Vein
                ...
                /ProjectionMaps
                    /OCTA(FULL)
                        10101.bmp
                        10102.bmp
                        ...
                    /OCTA(ILM_OPL)
                        10101.bmp
                        10102.bmp
                        ...
                    /OCTA(OPL_BM)
                        10101.bmp
                        10102.bmp
                        ...

Code path: funcs\dataset\octa_500.py

__ROSE__:

    /assets
        /datasets
            /gt
                /RV
                    01.png
                    02.png
                    ...
            /img
                /DVC
                    01.tif
                    02.tif
                    ...
                /IVC
                    01.png
                    02.png
                    ...
                /SVC
                    01.tif
                    02.tif
                    ...

Code path: funcs\dataset\rose_o.py

### 2.2 Key Code Paths

- Model Structure: funcs\model\DSCNet.py
- IBP Loss Function Definition: funcs\loss_function\clDice.py
- Training Hyperparameters: funcs\options\param_segment.py
- Training Framework: funcs\train\general_segment.py
- Start Training: train.py

### 2.3 Start Training

Install the packages in __requirements.txt__, then run:

    python train.py

During training, prediction images and metrics are saved by training date under: results/segmentation/train/yyyy_MM_dd_hh_mm_ss

### 2.4 Model Testing

Run the following to test the model:

    python evaluate.py

Testing instantiates the model first and then loads the weights, so set parameters according to the model spec. __m.pth__ denotes medium; set __layer_depth=3__, __feature_num=72__. __l.pth__ denotes large; set __layer_depth=4__, __feature_num=144__. Then place the weights at: assets\checkpoints\your_weight_name.pth. In __evaluate.py__, modify: para_manager.general_segment_args.weight_name = "your_weight_name"

Weights (Baidu Netdisk): Link: https://pan.baidu.com/s/1D7ShJYp0qPOQqBVvgZZ4Bg?pwd=q6sj Extraction code: q6sj

Result path: results/segmentation/evaluate/yyyy_MM_dd_hh_mm_ss

## 3. Segmentation Preview (OCTA-500)

- Retinal Vessels
![RV](./figures/RV_3M.gif)

- Artery
![Artery](./figures/Artery_3M.gif)


## 4. Others

Paper online: https://doi.org/10.1016/j.bspc.2025.109117
# S3-OCTA

一种用于光学相干断层扫描血管造影（OCTA）图像的视网膜血管分割的 __PyTorch__ 端到端深度学习框架。

## 1. 简要介绍

### 1.1 主要特点

- 以 __动态蛇型卷积 (DSConv)__ 作为骨干网络，专为管状血管结构设计。
- 基于欧拉特征的边界剥离进行 __中心线提取__。
- 利用提取的中心线设计的 __自定义损失函数__ 以增强训练效果。

### 1.2 数据集及相关论文

- __OCTA-500__: https://www.sciencedirect.com/science/article/abs/pii/S1361841524000173
- __ROSE__: https://ieeexplore.ieee.org/abstract/document/9284503
- __OCTA-25K__: https://arxiv.org/abs/2107.10476

### 1.3 显存
- **推荐**: 20GB GPU显存以获得最佳性能。
- ⚠️ **注意**: GPU显存少于3GB时性能会显著下降。


## 2. 模型训练

### 2.1 数据集组织形式：
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
                /FAZ
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

代码路径: funcs\dataset\octa_500.py

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

代码路径: funcs\dataset\rose_o.py

### 2.2 关键代码路径

- 模型结构: funcs\model\DSCNet.py
- IBP损失函数定义：funcs\loss_function\clDice.py 
- 训练超参数选项: funcs\options\param_segment.py
- 训练过程框架：funcs\train\general_segment.py
- 启动训练：launcher.py

### 2.3 启动训练

运行以下代码开始训练：

    python launcher.py

所需要的第三方库都是深度图像常用的，报错直接pip install 即可。

### 2.4 查看结果

训练过程中的预测图像和指标会根据训练日期保存于路径：results/segmentation/train/yyyy_MM_dd_hh_mm_ss

## 3.分割样本示例(OCTA-500)

- 视网膜血管
![RV](./figures/RV_3M.gif)

- 动脉
![Artery](./figures/Artery_3M.gif)

## 4. 其他

如果你觉得有用，请
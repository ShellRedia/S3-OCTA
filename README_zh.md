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

不同配置的模型所占用的显存的不同，可在 __train.py__ 文件中对模型深度和通道数进行修改：

- **推荐**: 12GB GPU。
- **最佳**: 20GB GPU显存以获得最佳性能。
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

- 模型结构: funcs\model\S3OCTA.py
- IBP损失函数定义：funcs\loss_function\clDice.py 
- 训练超参数选项: funcs\options\param_segment.py
- 训练过程框架：funcs\train\general_segment.py
- 启动训练：train.py

### 2.3 启动训练

依次安装 __requirements.txt__ 中的包，并运行以下文件开始训练：

    python train.py

训练过程中的预测图像和指标会根据训练日期保存于路径：results/segmentation/train/yyyy_MM_dd_hh_mm_ss

### 2.4 模型测试

运行以下文件对模型进行测试：

    python evaluate.py

测试的实现是先创建模型后再加载权重，所以需要根据模型规格设置对应参数。__m.pth__ 表示medium（中型），需设置 __layer_depth=3__, __feature_num=72__; __l.pth__ 表示large（大型），需设置 __layer_depth=4__, __feature_num=144__。随后将权重文件放在路径：assets\checkpoints\你的权重名.pth。并在 __evaluate.py__ 中, 修改语句para_manager.general_segment_args.weight_name = "你的权重名"

权重下载（百度网盘）：链接: https://pan.baidu.com/s/1D7ShJYp0qPOQqBVvgZZ4Bg?pwd=q6sj 提取码: q6sj 

预测图像和指标会根据训练日期保存于路径: results/segmentation/evaluate/yyyy_MM_dd_hh_mm_ss


## 3.分割样本示例(OCTA-500)

- 视网膜血管
![RV](./figures/RV_3M.gif)

- 动脉
![Artery](./figures/Artery_3M.gif)

## 4. 其他

论文已上线：https://doi.org/10.1016/j.bspc.2025.109117
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import os
from tqdm import tqdm

from funcs.options.param_segment import HyperParameterManager
from funcs.record.segmentation import ResultRecorder
from funcs.dataset.manager import get_datasets

from monai.networks.nets import *

from funcs.model.S3OCTA import S3OCTA

class EvaluationManager:
    def __init__(self, param_manager: HyperParameterManager):
        self.seg_args = param_manager.general_segment_args

        time_str = "_".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])

        print(time_str)
        print("model_name: ",self.seg_args.model_name)
        print("dataset: ",self.seg_args.dataset)
        print("label_type: ",self.seg_args.label_type)

        self.record_dir = "results/segmentation/evaluate/{}".format(time_str)
    
        self.recorder = ResultRecorder(param_manager, self.record_dir)

        dataset_train, dataset_val, dataset_test = get_datasets(self.seg_args.dataset, self.seg_args.label_type)

        self.train_loader  = DataLoader(dataset_train, batch_size=1)
        self.val_loader = DataLoader(dataset_val, batch_size=1)
        self.test_loader = DataLoader(dataset_test, batch_size=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_cuda = lambda x: x.to(torch.float).to(self.device)

        self.model = self.build_segmetation_model()
        self.model.load_state_dict(torch.load('assets/checkpoints/{}.pth'.format(self.seg_args.weight_name)))
        self.model.eval()

        def record_dataloader(dataloader, loader_type="val"):
            with torch.no_grad():
                for images, labels, sample_names in tqdm(dataloader, "metrics calculation", leave=False):
                    images, labels = map(self.to_cuda, [images, labels])
                    preds = self.model(images)
                    sample_name = str(sample_names[0].numpy())
                    self.recorder.save_prediction(image=images, label=labels, pred=preds, sub_dir=loader_type, sample_name=sample_name)
                    self.recorder.calculate_metrics(pred=preds, label=labels, prefix=loader_type+"-", sample_name=sample_name)
            
        record_dataloader(self.train_loader, loader_type="train")
        record_dataloader(self.val_loader, loader_type="val")
        record_dataloader(self.test_loader, loader_type="test")

        self.recorder.save_metric()

    def build_segmetation_model(self):
        mp = self.seg_args.model_params["S3OCTA"]

        model = S3OCTA(
            in_channels=3, 
            out_channels=1, 
            kernel_size=mp["kernel_size"],
            layer_depth=mp["layer_depth"],
            rate=mp["feature_num"],
            ga=mp["global_aggregation"]
        )
        return model.to(self.device)
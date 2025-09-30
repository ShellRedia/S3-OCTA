import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import os
import json
from tqdm import tqdm

from funcs.options.param_segment import HyperParameterManager
from funcs.record.segmentation import ResultRecorder
from funcs.dataset.manager import get_datasets
from funcs.model.DSCNet import DSCNet

from monai.networks.nets import *

from funcs.loss_function.general import DiceLoss
from funcs.loss_function.soft_skeleton_recall import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from funcs.loss_function.clDice import Soft_Dice_clDice, IBP_clDice

from funcs.optimizer.prodigy import Prodigy

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x
    
class TrainingManager:
    def __init__(self, param_manager: HyperParameterManager):
        self.train_args = param_manager.general_train_args
        self.seg_args = param_manager.general_segment_args

        time_str = "_".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
        self.record_dir = "results/segmentation/train/{}".format(time_str)
        self.cpt_dir = "{}/checkpoints".format(self.record_dir)
        os.makedirs(self.cpt_dir, exist_ok=True)

        fov = "6M" if self.seg_args.dataset=="OCTA-500_6M" else "3M"

        self.recorder = ResultRecorder(param_manager, self.record_dir)

        json_save_path = "{}/{}.json".format(self.record_dir, param_manager.get_basic_info())

        with open(json_save_path, "w") as json_file:
            json.dump(param_manager.get_ablation_info(), json_file)

        dataset_train, dataset_val, dataset_test = get_datasets(self.seg_args.dataset, self.seg_args.label_type)

        self.train_loader  = DataLoader(dataset_train, batch_size=self.train_args.batch_size)
        self.val_loader = DataLoader(dataset_val, batch_size=1)
        self.test_loader = DataLoader(dataset_test, batch_size=1)

        self.loss_func = {
            "DiceLoss": DiceLoss(),
            "SoftDiceLoss": SoftDiceLoss(smooth=1e-6),
            "MESoftDiceLoss": MemoryEfficientSoftDiceLoss(smooth=1e-6),
            "clDiceLoss": Soft_Dice_clDice(),
            "IBP_clDiceLoss": IBP_clDice(),
        }[self.seg_args.loss_func]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_cuda = lambda x: x.to(torch.float).to(self.device)

        self.model = self.build_segmetation_model()

        pg = [p for p in self.model.parameters() if p.requires_grad]

        self.scheduler = None
        self.optimizer = optim.AdamW(pg, lr=self.train_args.lr, weight_decay=1e-4)
        if self.train_args.optimizer == "prodigy":
            self.optimizer = Prodigy(pg, lr=1)
        elif self.train_args.optimizer == "decay":
            lr_lambda = lambda x: max(0.05, 0.98 ** x)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def build_segmetation_model(self):
        if self.seg_args.model_name == "SegResNet":
            model = SegResNet(in_channels=3, out_channels=1, spatial_dims=2, init_filters=64, blocks_down=(1, 2, 4, 4),blocks_up=(4, 4, 4))
            return ModifiedModel(model).to(self.device)
        elif self.seg_args.model_name == "UNet":
            model = UNet(in_channels=3, out_channels=1, spatial_dims=2, channels=[512, 512, 1024, 2048, 2048], strides=[2, 2, 2, 2], kernel_size=9)
            return ModifiedModel(model).to(self.device)
        elif self.seg_args.model_name == "FlexUNet":
            model = FlexUNet(in_channels=3, out_channels=1, spatial_dims=2, backbone="efficientnet-b8")
            return ModifiedModel(model).to(self.device)
        elif self.seg_args.model_name == "SwinUNETR":
            model = SwinUNETR(img_size=(512,512), in_channels=3, out_channels=1, feature_size=72, spatial_dims=2, use_v2=True)
            return ModifiedModel(model).to(self.device)
        elif self.seg_args.model_name == "DiNTS":
            dints_space = TopologyInstance(spatial_dims=2, num_blocks=12, device="cuda")
            model = DiNTS(dints_space=dints_space, in_channels=3, num_classes=1, spatial_dims=2)
            return ModifiedModel(model).to(self.device)
        elif self.seg_args.model_name == "DSCNet":
            mp = self.seg_args.model_params["DSCNet"]

            model = DSCNet(
                in_channels=3, 
                out_channels=1, 
                kernel_size=mp["kernel_size"],
                layer_depth=mp["layer_depth"],
                rate=mp["feature_num"],
                ga=mp["global_aggregation"]
            )
            return model.to(self.device)
            
    def train(self):
        self.record_performance(0)
        for epoch in tqdm(range(1, self.train_args.epochs+1), desc="training"):
            self.model.train()
            for images, labels, sample_names in tqdm(self.train_loader, "train batches", leave=False):
                images, labels = map(self.to_cuda, [images, labels])
                self.optimizer.zero_grad()
                preds = self.model(images)
                self.loss_func(preds, labels).backward()
                self.optimizer.step()
            if self.scheduler is not None: self.scheduler.step()
            if epoch % self.train_args.check_interval == 0: self.record_performance(epoch)

    def record_performance(self, epoch):
        self.model.eval()

        self.recorder.add_metric("epoch", epoch)
        self.recorder.add_metric("lr", self.optimizer.param_groups[0]['lr'])

        def record_dataloader(dataloader, loader_type="val", is_complete=True):
            with torch.no_grad():
                for images, labels, sample_ids in tqdm(dataloader, "metrics calculation", leave=False):
                    images, labels = map(self.to_cuda, [images, labels])
                    preds = self.model(images)

                    self.recorder.metric_average_dct[loader_type + "-loss"].append(self.loss_func(preds, labels).cpu().item()) # informal usage

                    if loader_type != "train": 
                        self.recorder.calculate_metrics(pred=preds, label=labels, prefix=loader_type+"-")

                    if is_complete:
                        self.recorder.save_prediction(image=images, label=labels, pred=preds, sub_dir="{:0>4}".format(epoch), sample_name=str(sample_ids[0].numpy()))
            if is_complete:
                torch.save(self.model, '{}/{}.pth'.format(self.cpt_dir, self.seg_args.model_name))

        is_complete = bool(epoch % (self.train_args.epochs // 4) == 0)
        record_dataloader(self.train_loader, loader_type="train", is_complete=is_complete)
        record_dataloader(self.val_loader, loader_type="val", is_complete=is_complete)
        record_dataloader(self.test_loader, loader_type="test", is_complete=is_complete)

        self.recorder.save_metric()

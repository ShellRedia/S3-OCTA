import os
import cv2
import torch
import numpy as np
import pandas as pd
import medpy.metric.binary as bm
from collections import defaultdict
from funcs.options.param_segment import HyperParameterManager
from funcs.record.custom_metrics import clDice
from imgaug import augmenters as iaa

class ResultRecorder:
    def __init__(self, param_manager: HyperParameterManager, record_dir: str, postprocess:bool=True, fov:str=""):
        self.record_dir = record_dir

        self.metric_dct = defaultdict(list)
        self.metric_average_dct = defaultdict(list)
        self.metric_sample_dct = defaultdict(list)

        self.seg_args = param_manager.general_segment_args
        self.calculate_metric_dct = {"Dice": bm.dc, "Jaccard": bm.jc, "clDice": clDice, "Hausdorff": bm.hd, "HD95": bm.hd95}

        self.postprocess = postprocess
        self.fov = fov

        if self.fov == "3M":
            self.augmenter = iaa.Resize({"height": 304, "width": 304})
        elif self.fov == "6M":
            self.augmenter = iaa.Resize({"height": 400, "width": 400})

        

    def calculate_metrics(self, pred, label, prefix="", sample_name=""):
        pred = torch.gt(pred, 0.8).int()
        label, pred = map(lambda x:x[0][0].cpu().detach().numpy(), [label, pred])


        if self.fov:
            label, pred = np.expand_dims(label.astype(np.uint8), axis=-1), np.expand_dims(pred.astype(np.uint8), axis=-1)
            _, mask = self.augmenter(images=[label, label], segmentation_maps=[label, pred])
            label, pred = mask[0][:,:,0] , mask[1][:,:,0] 

        if self.postprocess: pred = self.remove_small_components(pred, min_area=50, min_percent=10)

        if sample_name:
            self.metric_sample_dct["sample_name"].append(sample_name)
            self.metric_sample_dct["prefix"].append(prefix)
                
        for metric_name in self.seg_args.metrics:
            metric_func = self.calculate_metric_dct[metric_name]
            metric_value = metric_func(label, pred) if np.sum(pred) else 0
            self.metric_average_dct[prefix+metric_name].append(metric_value)
            if sample_name:
                self.metric_sample_dct[metric_name].append("{:.4f}".format(metric_value))


    def add_metric(self, metric_name, metric_value):
        self.metric_dct[metric_name].append(metric_value)


    def save_metric(self):
        for metric_name, metric_value_lst in self.metric_average_dct.items():
            self.add_metric(metric_name, "{:.4f}".format(round(np.mean(metric_value_lst), 4)))

        pd.DataFrame(self.metric_dct).to_excel("{}/metrics.xlsx".format(self.record_dir), index=False)
        if self.metric_sample_dct:
            sample_df = pd.DataFrame(self.metric_sample_dct)

            def save_subset(df, subset):
                subset_df = df[df["prefix"] == subset+"-"].drop(columns=['prefix'])
                subset_df.to_excel("{}/metrics_{}_sample.xlsx".format(self.record_dir, subset), index=False)

            for subset in ["train", "test", "val"]: save_subset(sample_df, subset)

        # clear
        self.metric_average_dct = defaultdict(list)
        self.metric_sample_dct = defaultdict(list)

    def save_prediction(self, image, label, pred, prompt_points=[], sub_dir="", sample_name="sample"):
        overlay = lambda x, y: cv2.addWeighted(x, 0.5, y, 0.5, 0)
        to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
        to_color = lambda x, color: (to_3ch(x) * color).astype(dtype=np.uint8)
        to_visible = lambda x : (x * 255 if x.max() <= 1 else x).astype(np.uint8)
        expand_dim = lambda x: np.expand_dims(x.astype(np.uint8), axis=-1)

        pred = torch.gt(pred, 0.8).int()
        image, label, pred = map(lambda x:to_visible(x[0][0].cpu().detach().numpy()), [image, label, pred])

        if self.fov:
            image, label, pred = map(expand_dim, [image, label, pred])
            image, mask = self.augmenter(images=[image, image], segmentation_maps=[label, pred])
            image, label, pred = image[0][:,:,0], mask[0][:,:,0] , mask[1][:,:,0] 

        image_gt = overlay(to_3ch(image), to_color(label, (0, 1, 0))) # green
        image_pred = overlay(to_3ch(image), to_color(pred, (0, 1, 1))) # yellow

        image_lst = [to_3ch(image), image_gt, image_pred]

        if prompt_points: 
            prompt_points = prompt_points.cpu().detach().numpy()
            image_pred_prompt = image_pred.copy()
            point_color_dct = {True:(0, 255, 0), False:(0, 0, 255)}

            for x, y, pn in prompt_points:
                cv2.circle(image_pred_prompt, (int(x), int(y)), 10, point_color_dct[pn], -1)
            image_lst.append(image_pred_prompt)

        save_dir = "{}/{}".format(self.record_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        cv2.imwrite("{}/{}.png".format(save_dir, sample_name), np.concatenate(image_lst, axis=1))

    def remove_small_components(self, input_image, min_area=8, min_percent=5):
        binary_image = (input_image > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        areas = stats[1:, cv2.CC_STAT_AREA] 

        if areas.size == 0:
            return np.zeros_like(binary_image, dtype=np.uint8)
        
        min_area_threshold = np.percentile(areas, min_percent)
        
        filtered_image = np.zeros_like(binary_image, dtype=np.uint8)
        for i in range(1, num_labels):  
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min(min_area, min_area_threshold):
                filtered_image[labels == i] = 1
                
        return filtered_image
    
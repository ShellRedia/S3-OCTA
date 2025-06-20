import os
import numpy as np
import cv2
import medpy.metric.binary as bm
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from funcs.record.custom_metrics import clDice

result_save_dir = "results/segmentation/train"
# result_dct = {
#     "2025_06_09_22_17_35":"UNet", # UNet_RV_3M
#     "2025_06_08_10_56_51":"SegRes", # SegRes_RV_3M
#     "2025_06_10_00_57_11":"DiNTS", # DiNTS_RV_3M
#     "2025_06_11_18_48_53":"SwinUNETR", # SwinUNETR_3M
#     "2025_06_07_17_37_56":"S3OCTA"  # S3OCTA_3M
# }

# result_dct = {
#     "2025_06_09_22_17_52":"UNet", # UNet_Artery_3M
#     "2025_06_08_10_58_57":"SegRes", # SegRes_Artery_3M
#     "2025_06_10_00_57_28":"DiNTS", # DiNTS_Artery_3M
#     "2025_06_11_18_17_34":"SwinUNETR", # SwinUNETR_Artery_3M
#     "2025_06_07_17_38_17":"S3OCTA"  # S3OCTA_Artery_3M
# }

# result_dct = {
#     "2025_06_09_22_18_14":"UNet", # UNet_Vein_3M
#     "2025_06_08_11_00_12":"SegRes", # SegRes_Vein_3M
#     "2025_06_10_09_22_54":"DiNTS", # DiNTS_Vein_3M
#     "2025_06_11_18_48_21":"SwinUNETR", # SwinUNETR_Vein_3M
#     "2025_06_08_10_14_40":"S3OCTA"  # S3OCTA_Vein_3M
# }

# result_dct = {
#     "2025_06_09_22_15_45":"UNet", # UNet_RV_6M
#     "2025_06_08_11_06_06":"SegRes", # SegRes_RV_6M
#     "2025_06_10_00_56_34":"DiNTS", # DiNTS_RV_6M
#     "2025_06_11_18_50_53":"SwinUNETR", # SwinUNETR_RV_6M
#     "2025_06_07_19_40_04":"S3OCTA"  # S3OCTA_RV_6M
# }

# result_dct = {
#     "2025_06_09_22_17_02":"UNet", # UNet_Artery_6M
#     "2025_06_08_13_36_38":"SegRes", # SegRes_Artery_6M
#     "2025_06_10_00_56_48":"DiNTS", # DiNTS_Artery_6M
#     "2025_06_11_18_51_20":"SwinUNETR", # SwinUNETR_Artery_6M
#     "2025_06_07_19_40_24":"S3OCTA"  # S3OCTA_Artery_6M
# }

# result_dct = {
#     "2025_06_09_21_19_19":"UNet", # UNet_Vein_6M
#     "2025_06_08_13_36_48":"SegRes", # SegRes_Vein_6M
#     "2025_06_10_00_57_01":"DiNTS", # DiNTS_Vein_6M
#     "2025_06_11_18_51_47":"SwinUNETR", # SwinUNETR_Vein_6M
#     "2025_06_07_18_32_19":"S3OCTA"  # S3OCTA_Vein_6M
# }

result_dct = {
    "2025_06_10_11_00_10":"UNet", # UNet_ROSE
    "2025_06_10_11_00_22":"SegRes", # SegRes_ROSE
    "2025_06_09_14_44_32":"DiNTS", # DiNTS_ROSE
    "2025_06_11_20_34_37":"SwinUNETR", # SwinUNETR_ROSE
    "2025_06_08_10_17_51":"S3OCTA"  # S3OCTA_ROSE
}



get_binary_mask = lambda x : np.where((lambda b, g, r: (r != g) | (r != b) | (g != b))(*cv2.split(x)), 1, 0).astype(np.uint8)

metric_func_dct = {"Dice": bm.dc, "Jaccard": bm.jc, "clDice": clDice, "HD95": bm.hd95}

g_sample, g_metric = defaultdict(list), defaultdict(list)

test_id_set = set(list(range(31, 40)) + list(range(10451, 10501)) + list(range(10201, 10301)))

is_test_sample = lambda x: bool(int(x[:-4]) in test_id_set)


for result_dir, dataset_name in result_dct.items():
    record_dir = "{}/{}/0200".format(result_save_dir, result_dir)
    for sample_file in tqdm(os.listdir(record_dir)):
        if is_test_sample(sample_file):
            sample_image = cv2.imread("{}/{}".format(record_dir, sample_file))
            h = sample_image.shape[0]
            image, label, pred = sample_image[:,:h], sample_image[:,h:-h], sample_image[:,-h:]
            if g_sample[sample_file]:
                g_sample[sample_file].append(pred)
            else:
                g_sample[sample_file] = [image, label, pred]

            sample_id = sample_file[:-4]
            if sample_id not in g_metric["sample_id"]: 
                g_metric["sample_id"].append(sample_id)
            
            for metric_name, metric_func in metric_func_dct.items():
                label_binary, pred_binary = map(get_binary_mask, [label, pred])
                g_metric["{}-{}".format(dataset_name, metric_name)].append("{:.4f}".format(metric_func(label_binary, pred_binary)))


save_dir = "sample_comparison"
os.makedirs(save_dir, exist_ok=True)

for sample_file, image_lst in g_sample.items():
    cv2.imwrite("{}/{}".format(save_dir, sample_file), np.concatenate(image_lst, axis=1))

pd.DataFrame(g_metric).to_excel("{}/metrics.xlsx".format(save_dir))
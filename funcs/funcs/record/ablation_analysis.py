import os
import json
import pandas as pd
from math import inf
from benedict import benedict
from collections import defaultdict

class AblationAnalysis:
    def __init__(self, 
                result_dir="results/segmentation/train", 
                model_name="DSCNet",
                label_type="RV",
                dataset="OCTA-500_6M",
                keyword="layer_depth",
                begin_date="",
                end_date=""
            ):

        comparison_items = ["val-loss", "test-loss"]

        best_value_dct, best_metric_dct = defaultdict(lambda: inf), defaultdict()

        self.ablation_info = "_".join([model_name, label_type, dataset])
        self.keyword = keyword

        json_file_name = self.ablation_info + ".json"

        for timestamp_dir in os.listdir(result_dir):
            if len(begin_date) and begin_date > timestamp_dir: continue
            if len(end_date) and end_date < timestamp_dir: continue

            result_record_dir = "{}/{}".format(result_dir, timestamp_dir)

            if json_file_name not in os.listdir(result_record_dir): continue

            with open("{}/{}".format(result_record_dir, json_file_name), 'r') as json_file:
                json_str = json_file.readline()
                config_dct = json.loads(json_str)
                b_dict = benedict(config_dct)
                selected_keyword = [x for x in b_dict.keypaths() if x.split('.')[-1] == keyword]

                if selected_keyword:
                    metric_df = pd.read_excel("{}/metrics.xlsx".format(result_record_dir))
                    comparison_item_value = b_dict[selected_keyword[0]]
                    for item in comparison_items:
                        comparison_key = "{}:{}".format(comparison_item_value, item)
                        if metric_df[item].min() < best_value_dct[comparison_key]:
                            best_value_dct[comparison_key] = metric_df[item].min()
                            best_metric_dct[comparison_key] = metric_df.loc[metric_df[item].idxmin()].to_dict()

        self.best_metric_dct = best_metric_dct
    
    def save_as_excel(self, metric_names=[]):
        save_dir = "analysis/abalation"
        os.makedirs(save_dir, exist_ok=True)
        if metric_names:
            for k in self.best_metric_dct:
                is_del = True
                for matric_name in metric_names:
                    if matric_name in k: is_del = False
                if is_del: del self.best_metric_dct[k]

        pd.DataFrame(self.best_metric_dct).T.to_excel('{}/{}_{}.xlsx'.format(save_dir, self.ablation_info, self.keyword), index=True)
            
if __name__=="__main__":
    aa = AblationAnalysis()
    aa.save_as_excel()
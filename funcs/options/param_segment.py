import json
from addict import Dict


class HyperParameterManager:
    def __init__(self):
        self.set_train_parameters()
        self.set_general_segment_parser()
        self.set_sam_parser()

    def get_basic_info(self):
        info_lst = [self.general_segment_args.model_name, self.general_segment_args.label_type, self.general_segment_args.dataset]
        return "_".join([str(x) for x in info_lst])
    
    def get_ablation_info(self):
        train_args, seg_args = self.general_train_args, self.general_segment_args
        albation_dct = {
            "model_name": seg_args.model_name,
            "loss_func": seg_args.loss_func,
            "optimizer": train_args.optimizer,
            "model_params": seg_args.model_params[seg_args.model_name],
            "training_epochs": train_args.epochs
        }
        
        return json.dumps(albation_dct)


    def set_train_parameters(self):
        self.general_train_args = Dict()
        self.general_train_args.device = "0"
        self.general_train_args.epochs = 200
        self.general_train_args.optimizer = "decay" # constant, decay, prodigy
        self.general_train_args.lr = 1e-4
        self.general_train_args.batch_size = 1
        self.general_train_args.check_interval = 5
    

    def set_general_segment_parser(self):
        self.general_segment_args = Dict()

        self.general_segment_args.model_params = {
            "DSCNet":{
                "extend_scope": 7,
                "kernel_size": 9,
                "layer_depth": 3,
                "feature_num": 72,
                "global_aggregation": "MiT" # SWinT, MiT, ASPP
            },
            "SegResNet":{},
            "UNet":{},
            "FlexUNet":{},
            "DiNTS":{},
            "SwinUNETR":{}
        }

        self.general_segment_args.dataset = "ROSE" # OCTA-500_3M_6M, OCTA-500_3M, OCTA-500_6M, ROSE
        self.general_segment_args.label_type = "RV" # RV, Artery, Vein
        self.general_segment_args.metrics = ["Dice", "Jaccard", "clDice", "HD95"]
        self.general_segment_args.model_name = "DSCNet_RV_Merge" # DSCNet, SegResNet, FlexUNet, DiNTS, UNet, SwinUNETR
        self.general_segment_args.loss_func = "IBP_clDiceLoss" # DiceLoss, clDiceLoss, IBP_clDiceLoss
        
    
    def set_sam_parser(self):
        self.sam_args = Dict()
        self.sam_args.rank = 16
        self.sam_args.prompt_positive_num = 6
        self.sam_args.prompt_negative_num = 6
        self.sam_args.model_type = "vit_l"
        self.sam_args.is_local = False
        self.sam_args.is_local = "LoRA"
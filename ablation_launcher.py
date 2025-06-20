
from funcs.train.general_segment import TrainingManager
from funcs.options.param_segment import HyperParameterManager


para_manager = HyperParameterManager()

para_manager.general_segment_args.label_type = "Vein" # RV, Artery, Vein

ablation_dct = {
    "model_name": {
        "DSCNet": {
            # "layer_depth" : [2,3,4,5] 
            "feature_num": [8, 24, 72] # 216
        }
    },
    "loss_func": ["DiceLoss", "clDiceLoss", "IBP_clDiceLoss"]
}

def exec_model_params():
    for model_name in ablation_dct["model_name"]:
        para_manager.general_segment_args.model_name = model_name
        model_param_dct = ablation_dct["model_name"][model_name]
        for model_param_name in model_param_dct:
            for param_value in model_param_dct[model_param_name]:
                para_manager.general_segment_args.model_params[model_name][model_param_name] = param_value
                TrainingManager(para_manager).train()

def exec_loss_func():
    for loss_func in ablation_dct["loss_func"]:
        para_manager.general_segment_args.loss_func = loss_func
        TrainingManager(para_manager).train()

# for dataset in "OCTA-500_3M", "OCTA-500_6M", "ROSE":
para_manager.general_segment_args.dataset = "OCTA-500_6M"
for label_type in "RV", "Artery", "Vein":
    para_manager.general_segment_args.label_type = label_type
    TrainingManager(para_manager).train()

# for dataset in "OCTA-500_6M", "ROSE":
#     para_manager.general_segment_args.dataset = dataset
#     exec_loss_func()
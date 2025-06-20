
from funcs.evaluate.general_segment import EvaluationManager
from funcs.options.param_segment import HyperParameterManager




para_manager = HyperParameterManager()



# for dataset in "OCTA-500_3M", "OCTA-500_6M":
#     para_manager.general_segment_args.dataset = dataset

#     for fov in "3M", "6M":
#         if fov not in dataset:
#             for label_type in "RV", "Artery", "Vein":
#                 para_manager.general_segment_args.label_type = label_type
#                 para_manager.general_segment_args.model_name = "DSCNet_{}_{}".format(label_type, fov)
#                 EvaluationManager(para_manager)

para_manager.general_segment_args.label_type = "RV"

for dataset in "OCTA-500_3M", "OCTA-500_6M", "ROSE":
    para_manager.general_segment_args.dataset = dataset

    for fov in "3M", "6M", "ROSE":
        if fov == "ROSE" or dataset == "ROSE" and fov != dataset:
            para_manager.general_segment_args.model_name = "DSCNet_RV_{}".format(fov)
            EvaluationManager(para_manager)
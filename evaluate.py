from funcs.evaluate.general_segment import EvaluationManager
from funcs.options.param_segment import HyperParameterManager

para_manager = HyperParameterManager()

para_manager.general_segment_args.model_params["S3OCTA"]["layer_depth"] = 3
para_manager.general_segment_args.model_params["S3OCTA"]["feature_num"] = 72

para_manager.general_segment_args.weight_name = "octa500_rv_3m_m"

EvaluationManager(para_manager)
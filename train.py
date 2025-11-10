from funcs.train.general_segment import TrainingManager
from funcs.options.param_segment import HyperParameterManager

para_manager = HyperParameterManager()
TrainingManager(para_manager).train()
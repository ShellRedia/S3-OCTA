from funcs.train.general_segment import TrainingManager
from funcs.evaluate.general_segment import EvaluationManager
from funcs.options.param_segment import HyperParameterManager

para_manager = HyperParameterManager()

# TrainingManager(para_manager).train()

EvaluationManager(para_manager)
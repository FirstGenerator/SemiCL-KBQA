from enum import Enum

from conf.setting import BartModelStrategy


class BaseData(Enum):
    source_ids = 'input_ids'
    source_mask = 'attention_mask'
    target_ids = 'decoder_input_ids'
    answers = 'answer'
    choices = 'choices'


class FuncClassData(Enum):
    source_ids = 'input_ids'
    source_mask = 'attention_mask'
    target_ids = 'decoder_input_ids'
    answers = 'answer'
    choices = 'choices'

    function_ids = 'decoder_func_ids'
    function_masks = 'func_masks'
    inputs_ids = 'decoder_inp_ids'
    input_masks = 'input_masks'


class ContrastiveLearningData(Enum):
    source_ids = 'input_ids'
    source_mask = 'attention_mask'
    target_ids = 'decoder_input_ids'
    answers = 'answer'
    choices = 'choices'

    tar_mask_ids = 'tar_mask_ids'


class TreeInfoData(Enum):
    source_ids = 'input_ids'
    source_mask = 'attention_mask'
    target_ids = 'decoder_input_ids'
    answers = 'answer'
    choices = 'choices'

    tree_data = 'tree_data'


DATA_MAP = {
    BartModelStrategy.CONTRASTIVE_LEARNING.value: ContrastiveLearningData,
    BartModelStrategy.PURE_CL.value: ContrastiveLearningData,
    BartModelStrategy.FUNC_CLASS.value: FuncClassData,
    BartModelStrategy.TREE_INFO.value: TreeInfoData,
}

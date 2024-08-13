import torch
from utils_basic import batchifier, get_rank, get_device_memory, compile_model, transpose_dict
from tqdm.auto import tqdm
from torch.nn.functional import softmax

def get_discourse_model_torch():
    """Get the Pytorch version of the discourse model.

    Returns dictionary with:
        { "model": <>, "tokenizer": <>, "config": <>, }
    """
    from transformers import AutoTokenizer, AlbertForSequenceClassification
    model_name = "alex2awesome/discourse-prediction__basic"
    discourse_tokenizer = AutoTokenizer.from_pretrained(model_name)
    discourse_model = AlbertForSequenceClassification.from_pretrained(model_name).eval()
    discourse_config = discourse_model.config
    for i in range(torch.cuda.device_count()):  # send model to every GPU
        discourse_model.to(f'cuda:{i}')
    # from torch.nn import DataParallel
    # discourse_model = DataParallel( discourse_model)
    # discourse_model = discourse_model.to('cuda:0')
    # discourse_model = compile_model(discourse_model, model_name)
    return {'model': discourse_model, 'tokenizer': discourse_tokenizer, 'config': discourse_config}

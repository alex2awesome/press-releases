import torch
from typing import List, Optional

label_mapper = {
    'Main': 0,
    'Main_Consequence': 1,
    'Cause_General': 2,
    'Cause_Specific': 3,
    'Distant_Anecdotal': 4,
    'Distant_Evaluation': 5,
    'Distant_Expectations_Consequences': 6,
    'Distant_Historical': 7,
    'Error': 8,
}

def _get_attention_mask(x: List[torch.Tensor], max_length_seq: Optional[int]=10000) -> torch.Tensor:
    max_len = max(map(lambda y: y.shape.numel(), x))
    max_len = min(max_len, max_length_seq)
    attention_masks = []
    for x_i in x:
        input_len = x_i.shape.numel()
        if input_len < max_length_seq:
            mask = torch.cat((torch.ones(input_len), torch.zeros(max_len - input_len)))
        else:
            mask = torch.ones(max_length_seq)
        attention_masks.append(mask)
    return torch.stack(attention_masks)
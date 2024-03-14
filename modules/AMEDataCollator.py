from transformers.data.data_collator import DefaultDataCollator
from collections.abc import Mapping
import numpy as np
import torch

def torch_AME_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {"return_loss": True}
    batch_main = {}

    keys = first.keys()

    for k in ("input_ids_1", "input_ids_2", "attention_mask_1", "attention_mask_2", "guid"):
        assert k in keys

    v = first["input_ids_1"]
    if v is not None and not isinstance(v, str):
        if isinstance(v, torch.Tensor):
            batch_main["input_ids"] = torch.stack([f["input_ids_1"] for f in features])
        elif isinstance(v, np.ndarray):
            batch_main["input_ids"] = torch.tensor(np.stack([f["input_ids_1"] for f in features]))
        else:
            batch_main["input_ids"] = torch.tensor([f["input_ids_1"] for f in features])
    
    v = first["attention_mask_1"]
    if v is not None and not isinstance(v, str):
        if isinstance(v, torch.Tensor):
            batch_main["attention_mask"] = torch.stack([f["attention_mask_1"] for f in features])
        elif isinstance(v, np.ndarray):
            batch_main["attention_mask"] = torch.tensor(np.stack([f["attention_mask_1"] for f in features]))
        else:
            batch_main["attention_mask"] = torch.tensor([f["attention_mask_1"] for f in features])
    
    v = first["input_ids_2"]
    if v is not None and not isinstance(v, str):
        if isinstance(v, torch.Tensor):
            batch_main["input_ids"] = torch.cat((batch_main["input_ids"], torch.stack([f["input_ids_2"] for f in features])), dim = 0)
        elif isinstance(v, np.ndarray):
            batch_main["input_ids"] = torch.cat((batch_main["input_ids"], torch.tensor(np.stack([f["input_ids_2"] for f in features]))), dim = 0)
        else:
            batch_main["input_ids"] = torch.cat((batch_main["input_ids"], torch.tensor([f["input_ids_2"] for f in features])), dim = 0)
    
    v = first["attention_mask_2"]
    if v is not None and not isinstance(v, str):
        if isinstance(v, torch.Tensor):
            batch_main["attention_mask"] = torch.cat((batch_main["attention_mask"], torch.stack([f["attention_mask_2"] for f in features])), dim = 0)
        elif isinstance(v, np.ndarray):
            batch_main["attention_mask"] = torch.cat((batch_main["attention_mask"], torch.tensor(np.stack([f["attention_mask_2"] for f in features]))), dim = 0)
        else:
            batch_main["attention_mask"] = torch.cat((batch_main["attention_mask"], torch.tensor([f["attention_mask_2"] for f in features])), dim = 0)

    if "embed_input_ids_1" in first:
        batch_1 = {}
        v = first["embed_input_ids_1"]
        if isinstance(v, torch.Tensor):
            batch_1["input_ids"] = torch.stack([f["embed_input_ids_1"] for f in features])
        elif isinstance(v, np.ndarray):
            batch_1["input_ids"] = torch.tensor(np.stack([f["embed_input_ids_1"] for f in features]))
        else:
            batch_1["input_ids"] = torch.tensor([f["embed_input_ids_1"] for f in features])
        
        v = first["embed_attention_mask_1"]
        if isinstance(v, torch.Tensor):
            batch_1["attention_mask"] = torch.stack([f["embed_attention_mask_1"] for f in features])
        elif isinstance(v, np.ndarray):
            batch_1["attention_mask"] = torch.tensor(np.stack([f["embed_attention_mask_1"] for f in features]))
        else:
            batch_1["attention_mask"] = torch.tensor([f["embed_attention_mask_1"] for f in features])
        batch["batch_1"] = batch_1
    
    if "embed_input_ids_2" in first:
        batch_2 = {}
        v = first["embed_input_ids_2"]
        if isinstance(v, torch.Tensor):
            batch_2["input_ids"] = torch.stack([f["embed_input_ids_2"] for f in features])
        elif isinstance(v, np.ndarray):
            batch_2["input_ids"] = torch.tensor(np.stack([f["embed_input_ids_2"] for f in features]))
        else:
            batch_2["input_ids"] = torch.tensor([f["embed_input_ids_2"] for f in features])
        
        v = first["embed_attention_mask_2"]
        if isinstance(v, torch.Tensor):
            batch_2["attention_mask"] = torch.stack([f["embed_attention_mask_2"] for f in features])
        elif isinstance(v, np.ndarray):
            batch_2["attention_mask"] = torch.tensor(np.stack([f["embed_attention_mask_2"] for f in features]))
        else:
            batch_2["attention_mask"] = torch.tensor([f["embed_attention_mask_2"] for f in features])
        batch["batch_2"] = batch_2
    batch["batch_main"] = batch_main

    return batch

class AMEDataCollator(DefaultDataCollator):
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors = None):
        # Only Pytorch is allowed
        # if return_tensors is None:
        #     return_tensors = self.return_tensors
        return torch_AME_data_collator(features)
import torch
import random
from torch.utils.data import Sampler
from transformers import logging
logger = logging.get_logger(__name__)

def get_length_in_batch_grouped_indices(lengths, one_batch_size, generator=None):
    idxs = torch.randperm(len(lengths), generator=generator).tolist()

    ret = []
    for i in range(len(idxs) // one_batch_size):
        ret.extend(idxs[i * one_batch_size:(i + 1) * one_batch_size])

    return ret

def get_length_in_batch_grouped_indices_idx_dict(lengths, world_size, one_batch_size, idx_dict = None, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    def shuffle_partially(idxs, start, end, padding_end):
        randalloc = torch.randint(start, end, (padding_end - end, ), generator=generator)
        idxs[end:padding_end] = randalloc
        shuf = torch.randperm(padding_end - start, generator=generator) + start
        idxs[start:padding_end] = idxs[shuf]
        return idxs

    if idx_dict:
        idxs = torch.arange(len(lengths))
        for _, idx_lang in idx_dict.items():
            idxs = shuffle_partially(idxs, idx_lang[0], idx_lang[1], idx_lang[2])
        idxs = idxs.tolist()
        ret_list = []
        batch_size = world_size * one_batch_size
        for i in range(len(idxs) // batch_size):
            ret_list.append(idxs[i * batch_size:(i + 1) * batch_size])
        random.shuffle(ret_list)
        ret = []
        for tmp in ret_list:
            ret.extend(tmp)
        return ret
    else:
        get_length_in_batch_grouped_indices(lengths, one_batch_size, generator)

class LengthInBatchSampler(Sampler):
    def __init__(
        self,
        batch_size,
        one_batch_size,
        dataset = None,
        lengths = None,
        model_input_name = None,
        generator = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            lengths = [256] * len(dataset)
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthCrossAcceleratorSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        if one_batch_size % 2 != 0:
            raise ValueError(
                'Batch size should be multiple of 2'
            )
        self.one_batch_size = one_batch_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_in_batch_grouped_indices(self.lengths, self.one_batch_size, generator=self.generator)
        return iter(indices)

class LengthInBatchwithIdxDictSampler(Sampler):

    def __init__(
        self,
        batch_size,
        one_batch_size,
        world_size,
        idx_dict,
        dataset = None,
        lengths = None,
        model_input_name = None,
        generator = None,

    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            lengths = [256] * len(dataset)
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthCrossAcceleratorSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        if one_batch_size % 2 != 0:
            raise ValueError(
                'Batch size should be multiple of 2'
            )
        
        self.one_batch_size = one_batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.idx_dict = idx_dict
        random.seed(self.generator.initial_seed())

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_in_batch_grouped_indices_idx_dict(self.lengths, self.world_size, self.one_batch_size, self.idx_dict, generator=self.generator)
        return iter(indices)
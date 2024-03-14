import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler

import datasets

from .AMESampler import LengthInBatchSampler, LengthInBatchwithIdxDictSampler

from transformers import AutoModel, AutoConfig
from transformers import Trainer
from transformers.utils import is_datasets_available
from transformers import logging
logger = logging.get_logger(__name__)

def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False

class AMETrainer(Trainer):
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
    ):
        super().__init__(
            model, 
            args, 
            data_collator, 
            train_dataset, 
            eval_dataset, 
            tokenizer, 
            model_init, 
            compute_metrics, 
            callbacks, 
            optimizers, 
            preprocess_logits_for_metrics
        )
        
        if self.args.embed_model_dir == "self":
            self.embed_model = model
        elif self.args.embed_model_dir == None:
            self.embed_model = None
        else:
            embed_config = AutoConfig.from_pretrained(self.args.embed_model_dir, output_hidden_states=True, output_past=False)
            self.embed_model = AutoModel.from_pretrained(self.args.embed_model_dir, config=embed_config, add_pooling_layer = False).to(self.args.device)
            self.embed_model.eval()

    
    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = "input_ids_1"
            assert self.args.train_batch_size % 2 == 0
            g = torch.Generator()
            g.manual_seed(self.args.seed)
            return LengthInBatchwithIdxDictSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                self.args.train_batch_size,
                self.args.world_size,
                self.args.dataset_sentence_idx_dict,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=g
            )
        else:
            return RandomSampler(self.train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        if eval_dataset is None or not has_length(eval_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
                lengths = (
                    eval_dataset[self.args.length_column_name]
                    if self.args.length_column_name in eval_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = "input_ids_1"
            return LengthInBatchSampler(
                self.args.eval_batch_size * self.args.gradient_accumulation_steps,
                self.args.eval_batch_size,
                dataset=eval_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(eval_dataset)
    
    def get_pseudo_mask_eye(self):
        if self.args.cross_acc_sampling:
            pseudo_labels = torch.eye(self.args.per_device_train_batch_size * self.args.world_size, self.args.per_device_train_batch_size * self.args.world_size).to(self.args.device)
        else:
            pseudo_labels = torch.eye(self.args.per_device_train_batch_size, self.args.per_device_train_batch_size).to(self.args.device)
        return pseudo_labels
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        AME custom loss to utilize pseudo-label
        """
        cur_train = model.training
        # pseudo label generating
        with torch.no_grad():
            if self.embed_model:
                self.embed_model.eval()
                if "batch_1" in inputs and "batch_2" in inputs:
                    pseudo_label_embed_1 = self._make_embed(self.embed_model, inputs["batch_1"])
                    pseudo_label_embed_2 = self._make_embed(self.embed_model, inputs["batch_2"])
                    pseudo_labels_1 = self._compute_sim_matrix(pseudo_label_embed_1, pseudo_label_embed_1)
                    pseudo_labels_2 = self._compute_sim_matrix(pseudo_label_embed_2, pseudo_label_embed_2)
                    pseudo_labels = (pseudo_labels_1 + pseudo_labels_2) / 2
                elif "batch_1" in inputs:
                    pseudo_label_embed = self._make_embed(self.embed_model, inputs["batch_1"])
                    pseudo_labels = self._compute_sim_matrix(pseudo_label_embed, pseudo_label_embed)
                elif "batch_2" in inputs:
                    pseudo_label_embed = self._make_embed(self.embed_model, inputs["batch_2"])
                    pseudo_labels = self._compute_sim_matrix(pseudo_label_embed, pseudo_label_embed)
                else:
                    pseudo_labels = self.get_pseudo_mask_eye()
            else:
                pseudo_labels = self.get_pseudo_mask_eye()
        # similarity matrix generating
        if cur_train:
            model.train()
        else:
            model.eval()
        embeds_1, embeds_2, outputs = self._make_embed(model, inputs["batch_main"], train=True, return_outputs=True)
        sim = self._compute_sim_matrix(embeds_1, embeds_2, train=True)

        # AME Loss
        assert sim.size() == pseudo_labels.size()

        loss = self._compute_loss_from_type(sim, pseudo_labels) + self._compute_loss_from_type(sim.t(), pseudo_labels.t())
        if self.args.train_mono_space:
            sim_1 = self._compute_sim_matrix(embeds_1, embeds_1, train=True)
            sim_2 = self._compute_sim_matrix(embeds_2, embeds_2, train=True)
            loss += self._compute_loss_from_type(sim_1, pseudo_labels) + self._compute_loss_from_type(sim_1.t(), pseudo_labels.t())
            loss += self._compute_loss_from_type(sim_2, pseudo_labels) + self._compute_loss_from_type(sim_2.t(), pseudo_labels.t())
        return (loss, outputs) if return_outputs else loss
    
    def _compute_sim_matrix(self, embed_1, embed_2, train = False):
        if train:
            ret = torch.mm(embed_1, embed_2.t())
        else:
            with torch.no_grad():
                ret = torch.mm(embed_1, embed_2.t())
        return ret
    
    def _encode(self, model, input):
        output = model(**input)
        token_embeddings = output["last_hidden_state"]
        input_mask_expanded = input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeds = F.normalize(embeds, p = 2, dim = 1)
        output.last_hidden_state = embeds.view(2, embeds.size(0) // 2, embeds.size(1)).permute(1, 0, 2)
        return output, embeds

    def _encode_embed(self, model, input):
        return self._encode(model, input)

    def _make_embed(self, model, input, train = False, return_outputs = False):
        if train:
            output, embeds = self._encode(model, input)
            batch_size = embeds.size(0) // 2
            embeds_1 = embeds[:batch_size]
            embeds_2 = embeds[batch_size:]
            if self.args.cross_acc_sampling:
                gathered_embeds_1 = [torch.empty_like(embeds_1, device=self.args.device) for _ in range(self.args.world_size)]
                gathered_embeds_2 = [torch.empty_like(embeds_2, device=self.args.device) for _ in range(self.args.world_size)]
                dist.all_gather(gathered_embeds_1, embeds_1)
                dist.all_gather(gathered_embeds_2, embeds_2)
                gathered_embeds_1[self.args.local_rank] = embeds_1
                gathered_embeds_2[self.args.local_rank] = embeds_2
                gathered_embeds_1 = torch.cat(gathered_embeds_1)
                gathered_embeds_2 = torch.cat(gathered_embeds_2)
            else:
                gathered_embeds_1 = embeds_1
                gathered_embeds_2 = embeds_2
            if return_outputs:
                return gathered_embeds_1, gathered_embeds_2, output
            else:
                return gathered_embeds_1, gathered_embeds_2
        else:
            with torch.no_grad():
                output, embeds = self._encode_embed(model, input)
                if self.args.cross_acc_sampling:
                    gathered_embeds = [torch.empty_like(embeds, device=self.args.device) for _ in range(self.args.world_size)]
                    dist.all_gather(gathered_embeds, embeds)
                    gathered_embeds = torch.cat(gathered_embeds)
                else:
                    gathered_embeds = embeds
        if return_outputs:
            return gathered_embeds, output
        else:
            return gathered_embeds

    def _compute_loss_from_type(self, sim, pseudo_labels):
        with torch.no_grad():
            positive_mask = pseudo_labels > self.args.positive_threshold
            w = self._w_func(pseudo_labels, positive_mask)
        if self.args.loss_type == "CS":
            sim_mat = self._w_softmax(sim, positive_mask, self.args.tau, self.args.margin)
            batch_size = sim_mat.size(0)
            loss = - torch.sum(w * torch.log(sim_mat)) / batch_size
        elif self.args.loss_type == "MMSE":
            loss = torch.mean(torch.pow((sim - w) / self.args.tau, 2))
        else:
            raise ValueError("loss type is wrong, you should use CS or MCS")
        return loss

    def _w_func(self, sim_matrix, positive_mask):
        if self.args.w_func == "Softmax":
            return self._w_softmax(sim_matrix, positive_mask, self.args.tau, 0)
        elif self.args.w_func == "raw":
            return sim_matrix
        else:
            raise ValueError("w func is wrong, you should use Softmax or raw")
    
    def _w_softmax(self, sim_matrix, positive_mask, tau, margin):
        exps = torch.exp((sim_matrix - positive_mask * margin) / tau)

        if self.args.small_batch_size:
            total_batch_size = self.args.train_batch_size * self.args.world_size if self.args.cross_acc_sampling else self.args.train_batch_size
            assert total_batch_size % self.args.small_batch_size == 0
            batch_num = total_batch_size // self.args.small_batch_size
            view_size = (total_batch_size, batch_num, self.args.small_batch_size)
            w = (exps.view(view_size) / torch.sum(exps.view(view_size), dim = -1, keepdim=True)).view(total_batch_size, total_batch_size)
        else:
            w = exps / torch.sum(exps, dim = 1, keepdim=True)
        return w
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from datasets import Dataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mteb import MTEB

from tqdm import tqdm

from mteb_arguments import *

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def _transform_func(tokenizer, examples, args):
    if args.prompt:
        examples['input_texts'] = [args.prompt + t for t in examples['input_texts']]
    batch_dict = tokenizer(examples['input_texts'], max_length=512, padding="max_length", truncation=True)

    return batch_dict

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.args = args
        self.gpu_count = torch.cuda.device_count()

        self.model.eval()
        self.model.cuda()

        if self.gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _make_embed(self, model, input):
        output = model(**input)
        token_embeddings = output["last_hidden_state"]
        input_mask_expanded = input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float().to(token_embeddings.device)
        embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # embeds = F.normalize(embeds, p = 2, dim = 1)
        return embeds
        
    @torch.no_grad()
    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(lambda examples:_transform_func(tokenizer = self.tokenizer, examples = examples, args=self.args))
        
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)

        embed_size = 1024 if self.args.tokenizer == "xlm-roberta-large" else 768

        # ** use the batch size from args! **
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2 * self.gpu_count,
            collate_fn=data_collator,
            pin_memory=True)
        
        result_data = torch.empty((0, embed_size))
        for batch_dict in tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                embed = self._make_embed(self.model, batch_dict).detach().cpu()
                result_data = torch.cat((result_data, embed), dim=0)
        return result_data.numpy()

class LASER2_encoder():
    def __init__(self, args):
        self.args=args
        downloader = LaserModelDownloader(model_dir="/LASER2/")
        downloader.download_laser2()

        self.encoder = initialize_encoder(lang = "eng_Latn", model_dir="/LASER2/")
        self.tokenizer = initialize_tokenizer(lang = "eng_Latn", model_dir="/LASER2/")
    
    def encode(self, sentences, batch_size=32, **kwargs):
        tokenized_sentence = self.tokenizer(sentences)
        return self.encoder.encode_sentences(tokenized_sentence)

def main():
    args = parse()
    if args.model_name == "LASER2":
        from laser_encoders import initialize_encoder, initialize_tokenizer
        from laser_encoders.download_models import LaserModelDownloader
        model = LASER2_encoder(args)
    else:
        model = Model(args)

    print("Run MTEB on Model :", args.model_name)
    print()

    task_langs_ = [t for t in args.langs if t.strip()]

    if args.evaluation_task:
        evaluation = MTEB(
            tasks = args.evaluation_task.split(" "),
            task_langs = task_langs_
        )

    else:
        args.task_types = [t for t in args.task_types if t.strip()]
        evaluation = MTEB(
            task_types=args.task_types or None,
            task_langs=task_langs_
        )

    results = evaluation.run(model, output_folder=f"results/{args.model_name}", verbosity=2)
    print(results)

main()
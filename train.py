import argparse

from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset, concatenate_datasets
from modules.AMETrainer import AMETrainer
from modules.AMEDataCollator import AMEDataCollator
from modules.AMETrainingArguments import AMETrainingArguments
from modules.AMEMetric import AMEMetric, RemoveBadSaveCallback
from huggingface_hub import HfFileSystem

# torch TF32 Setting
import torch
torch.backends.cuda.matmul.allow_tf32 = True

def parse():
    parser = argparse.ArgumentParser()

    # From here, path arguments
    parser.add_argument(
        "--run_name",
        type = str,
        default = "default_run_name",
        help = "Run name"
    )
    parser.add_argument(
        "--continue_from_last_ckpt",
        action = "store_true",
        help = "Continue from the last ckpt in the output dir"
    )
    parser.add_argument(
        "--dataset_sentence",
        type = str,
        default = "wecover/OPUS",
        help = "Train dataset path, should follow our folder and file format just like https://huggingface.co/datasets/wecover/OPUS/tree/main"
    )
    parser.add_argument(
        "--dataset_eval",
        type = str,
        default = "wecover/OPUS_Tatoeba",
        help = "Validation dataset path"
    )
    parser.add_argument(
        "--langs",
        nargs='+', 
        default=["en", "ko"], 
        help='Languages to train, should be included at dataset_sentence and dataset_eval'
    )
    parser.add_argument(
        "--embed_model",
        type = str,
        default = None,
        help = "Teacher model path, type self if you want to use the training model and type None to use conventional contrastive learning"
    )
    parser.add_argument(
        "--embed_direction",
        type = str,
        default = "input",
        choices = ["left", "right", "both", "input"],
        help = "Embedding direction, left/right/both/input, if input, --langs input order will be the priority, place higher priority at first to use this"
    )
    parser.add_argument(
        "--train_mono_space",
        action = "store_true",
        help = "Also train monolingual space through contrastive learning"
    )
    parser.add_argument(
        "--model",
        type = str,
        default = "intfloat/multilingual-e5-base",
        help = "Student model path"
    )
    parser.add_argument(
        "--output",
        type = str,
        default = "./output/",
        help = "Output path"
    )
    parser.add_argument(
        "--log_dir",
        type = str,
        default = "./logs",
        help = "Logging path"
    )
    parser.add_argument(
        "--keep_all_ckpt",
        action = "store_true",
        help = "keep all ckpts. if false, remove useless ckpts, that are not best ckpts."
    )
    parser.add_argument(
        "--metrics_for_save_model",
        nargs = '+', 
        default = ("loss", "tatoeba_acc_avg", "step"), 
        help = "Save best model for each of the metric"
    )
    parser.add_argument(
        "--metrics_larger_better",
        nargs = '+',
        default = (False, True, True),
        help = "This is to define the best model for metrics_for_save_model, True means that the larger score the better"
    )

    # From here, hyperparameter arguments
    parser.add_argument(
        "--cross_acc_sampling",
        action ="store_true",
        help = "Enable cross accelerator samplling"
    )
    parser.add_argument(
        "--small_batch_size",
        type = int,
        default = None,
        help = "Divide similar matrix into (small_batch_size x small_batch_size) when computing loss, we recommand not to use this"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type = int,
        default = 14,
        help = "Batch size per device"
    )
    parser.add_argument(
        "--margin",
        type = float,
        default = 0.0,
        help = "margin of positive pair in contrastive learning"
    )
    parser.add_argument(
        "--positive_threshold",
        type = float,
        default = 0.95,
        help = "threshold for pseudo-label to predict as positive pair in contrastive learning, this only works at margin != 0"
    )
    parser.add_argument(
        "--loss_type",
        type = str,
        default = "CS",
        choices = ["CS", "MMSE"],
        help = "Loss type, CS means Contrastive learning using Soft label and MMSE means Multilingual MSE distillation"
    )
    parser.add_argument(
        "--tau",
        type = float,
        default = 0.1,
        help = "Temperature to use at contrastive learning"
    )
    parser.add_argument(
        "--w_func",
        type = str,
        default = "raw",
        choices=["Softmax", "raw"],
        help = "Weight fucntion to forward pseudo-label"
    )
    parser.add_argument(
        "--epoch",
        type = int,
        default = 30,
        help = "Train epoch"
    )
    parser.add_argument(
        "--eval_steps",
        type = int,
        default = 10000,
        help = "Evaluation, save steps"
    )
    parser.add_argument(
        "--max_steps",
        type = int,
        default = -1,
        help = "Train max steps, this overrides epoch."
    )
    parser.add_argument(
        "--lr",
        type = float,
        default = 3e-5,
        help = "Train learning rate"
    )

    # below is GPU arguments
    parser.add_argument( # Not using these, just to avoid errors
        "--local_rank",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--local-rank",
        type = int,
        default = 0
    )
    args = parser.parse_args()
    args.world_size = torch.cuda.device_count()
    args.lang2priority = {}
    for priority, lang in enumerate(args.langs):
        args.lang2priority[lang] = priority
    print("priority :", args.lang2priority)
    return args

def download_process_dataset(args, dataset_name, split):
    fs = HfFileSystem()

    args.langs = sorted(args.langs, reverse=True)
    print("Language :", args.langs)
    langs_dict = {}
    dataset_path = "datasets/"+dataset_name+"/"
    print("Checking", dataset_path)
    if split == "train":
        filename = "/train.parquet"
    elif split == "valid":
        filename = "/valid.parquet"
    else:
        ValueError("only train and valid are allowed in process_dataset")
    for idx1 in range(len(args.langs)):
        for idx2 in range(idx1):
            if idx1 != idx2:
                lang1 = args.langs[idx1]
                lang2 = args.langs[idx2]
                corpus_list = fs.glob(dataset_path+"*/"+lang1+'-'+lang2+"/")
                print(lang1+'.'+lang2)
                for corpus in corpus_list:
                    if len(fs.glob(corpus + filename)) != 0:
                        print(corpus[len(dataset_path):])
                        if lang1+'.'+lang2 in langs_dict:
                            langs_dict[lang1+'.'+lang2].append(corpus[len(dataset_path):] + filename)
                        else:
                            langs_dict[lang1+'.'+lang2] = [corpus[len(dataset_path):] + filename]
    dataset_dict = load_dataset(dataset_name, data_files=langs_dict)
    langs_dict = {}
    dataset_list = list(dataset_dict.items())

    dataset = dataset_list[0][1]
    start = 0
    end = len(dataset)
    if split == "train":
        mod = len(dataset) % (args.per_device_batch_size * args.world_size)
        if mod:
            for _ in range(args.per_device_batch_size * args.world_size - mod):
                dataset = dataset.add_item({"sentence1": "", "sentence2": "", "lang1": "", "lang2": "", "guid": -1})
        pad = len(dataset)
        dataset_idx_dict = {dataset_list[0][0] : (start, end, pad)}

        for lang, data in dataset_list[1:]:
            start = len(dataset)
            dataset = concatenate_datasets([dataset, data])
            end = len(dataset)
            mod = len(dataset) % (args.per_device_batch_size * args.world_size)
            if mod:
                for _ in range(args.per_device_batch_size * args.world_size - mod):
                    dataset = dataset.add_item({"sentence1": "", "sentence2": "", "lang1": "", "lang2": "", "guid": -1})
            pad = len(dataset)
            dataset_idx_dict[lang] = (start, end, pad)
        return dataset, dataset_idx_dict
    else:
        return dataset_dict

def prepare_trainer_input(args):
    dataset_sentence, dataset_sentence_idx_dict = download_process_dataset(args, args.dataset_sentence, "train")
    dataset_eval = download_process_dataset(args, args.dataset_eval, "valid")

    config = AutoConfig.from_pretrained(args.model, output_hidden_states=False, output_past=False)
    model = AutoModel.from_pretrained(args.model, config=config, add_pooling_layer = False)

    runname = args.run_name
    
    training_args = AMETrainingArguments(
        # path parameters
        output_dir = args.output,
        cross_acc_sampling = args.cross_acc_sampling,
        logging_dir = args.log_dir,
        # teacher embedding parameters
        embed_model_dir = args.embed_model,
        embed_direction = args.embed_direction,
        train_mono_space = args.train_mono_space,
        # essential parameters
        auto_find_batch_size = False,
        remove_unused_columns = False,
        group_by_length = True,
        gradient_accumulation_steps = 1,
        prediction_loss_only = False,
        do_train = True,
        do_eval = True,
        ddp_find_unused_parameters = False, 
        dataloader_num_workers = 8,
        # disable_tqdm = True,
        # hyperparameters
        num_train_epochs = args.epoch,
        max_steps = args.max_steps,
        learning_rate = args.lr,
        per_device_train_batch_size = args.per_device_batch_size, # actual GPU usage is twice
        per_device_eval_batch_size = args.per_device_batch_size, # actual GPU usage is twice
        dataset_sentence_idx_dict = dataset_sentence_idx_dict,
        small_batch_size = args.small_batch_size,
        loss_type = args.loss_type,
        margin = args.margin,
        positive_threshold = args.positive_threshold,
        tau = args.tau,
        w_func = args.w_func,
        # options for mixed precision, amp
        tf32 = True,
        fp16 = True,
        # save or eval parameters
        save_strategy = "steps",
        evaluation_strategy = "steps",
        logging_strategy = "steps",
        save_steps = args.eval_steps,
        eval_steps = args.eval_steps,
        logging_steps = args.eval_steps,

        # this is metric"s" for save model!
        metrics_for_save_model = args.metrics_for_save_model,
        metrics_larger_better = args.metrics_larger_better,
        log_level = 'info',
        report_to = "wandb",
        run_name = runname,
        seed = 7777777
    )
    collator = AMEDataCollator()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.embed_model:
        if args.embed_model == "self":
            tokenizer_embed = tokenizer
        else:
            tokenizer_embed = AutoTokenizer.from_pretrained(args.embed_model)
    
    def tokenize_function(examples):
        sentence1_encoding = tokenizer(examples["sentence1"], padding="max_length", truncation=True, max_length = 512)
        sentence2_encoding = tokenizer(examples["sentence2"], padding="max_length", truncation=True, max_length = 512)

        if args.embed_model:
            if "e5" in args.embed_model:
                examples["sentence1"] = ["query: " + text for text in examples["sentence1"]]
                examples["sentence2"] = ["query: " + text for text in examples["sentence2"]]
            if args.embed_direction == "both":
                embed_sentence1_encoding = tokenizer_embed(examples["sentence1"], padding="max_length", truncation=True, max_length = 512)
                embed_sentence2_encoding = tokenizer_embed(examples["sentence2"], padding="max_length", truncation=True, max_length = 512)
                return {
                    "input_ids_1": sentence1_encoding["input_ids"],
                    "input_ids_2": sentence2_encoding["input_ids"],
                    "attention_mask_1": sentence1_encoding["attention_mask"],
                    "attention_mask_2": sentence2_encoding["attention_mask"],
                    "embed_input_ids_1": embed_sentence1_encoding["input_ids"],
                    "embed_input_ids_2": embed_sentence2_encoding["input_ids"],
                    "embed_attention_mask_1": embed_sentence1_encoding["attention_mask"],
                    "embed_attention_mask_2": embed_sentence2_encoding["attention_mask"],
                }
            elif args.embed_direction == "left":
                embed_sentence1_encoding = tokenizer_embed(examples["sentence1"], padding="max_length", truncation=True, max_length = 512)
                return {
                    "input_ids_1": sentence1_encoding["input_ids"],
                    "input_ids_2": sentence2_encoding["input_ids"],
                    "attention_mask_1": sentence1_encoding["attention_mask"],
                    "attention_mask_2": sentence2_encoding["attention_mask"],
                    "embed_input_ids_1": embed_sentence1_encoding["input_ids"],
                    "embed_attention_mask_1": embed_sentence1_encoding["attention_mask"],
                }
            elif args.embed_direction == "right":
                embed_sentence2_encoding = tokenizer_embed(examples["sentence2"], padding="max_length", truncation=True, max_length = 512)
                return {
                    "input_ids_1": sentence1_encoding["input_ids"],
                    "input_ids_2": sentence2_encoding["input_ids"],
                    "attention_mask_1": sentence1_encoding["attention_mask"],
                    "attention_mask_2": sentence2_encoding["attention_mask"],
                    "embed_input_ids_2": embed_sentence2_encoding["input_ids"],
                    "embed_attention_mask_2": embed_sentence2_encoding["attention_mask"],
                }
            elif args.embed_direction == "input":
                embed_sentence1_encoding_tmp = tokenizer_embed(examples["sentence1"], padding="max_length", truncation=True, max_length = 512)
                embed_sentence2_encoding_tmp = tokenizer_embed(examples["sentence2"], padding="max_length", truncation=True, max_length = 512)
                embed_sentence1_encoding = {"input_ids" : [], "attention_mask" : []}
                for idx in range(len(examples["lang1"])):
                    # Only same languages can be tokenized through slicing
                    if (examples["lang1"][idx] == "" or examples["lang2"][idx] == ""):
                        embed_sentence1_encoding["input_ids"].append(embed_sentence1_encoding_tmp["input_ids"][idx])
                        embed_sentence1_encoding["attention_mask"].append(embed_sentence1_encoding_tmp["attention_mask"][idx])
                    else:
                        if args.lang2priority[examples["lang1"][idx]] <= args.lang2priority[examples["lang2"][idx]]:
                            embed_sentence1_encoding["input_ids"].append(embed_sentence1_encoding_tmp["input_ids"][idx])
                            embed_sentence1_encoding["attention_mask"].append(embed_sentence1_encoding_tmp["attention_mask"][idx])
                        else:
                            embed_sentence1_encoding["input_ids"].append(embed_sentence2_encoding_tmp["input_ids"][idx])
                            embed_sentence1_encoding["attention_mask"].append(embed_sentence2_encoding_tmp["attention_mask"][idx])
                return {
                    "input_ids_1": sentence1_encoding["input_ids"],
                    "input_ids_2": sentence2_encoding["input_ids"],
                    "attention_mask_1": sentence1_encoding["attention_mask"],
                    "attention_mask_2": sentence2_encoding["attention_mask"],
                    "embed_input_ids_1": embed_sentence1_encoding["input_ids"],
                    "embed_attention_mask_1": embed_sentence1_encoding["attention_mask"],
                }
            else:
                raise ValueError("embed_direction value error")
        else:
            return {
                "input_ids_1": sentence1_encoding["input_ids"],
                "input_ids_2": sentence2_encoding["input_ids"],
                "attention_mask_1": sentence1_encoding["attention_mask"],
                "attention_mask_2": sentence2_encoding["attention_mask"]
            }

    dataset_sentence = dataset_sentence.map(tokenize_function, batched=True, num_proc = 32)
    dataset_eval = dataset_eval.map(tokenize_function, batched=True, num_proc = 32)
    return model, training_args, dataset_sentence, dataset_eval, collator

def main():
    args = parse()
    model, training_args, dataset_sentence, dataset_eval, collator = prepare_trainer_input(args)
    trainer = AMETrainer(
        model = model,
        args = training_args, 
        data_collator = collator, 
        train_dataset = dataset_sentence, 
        eval_dataset = dataset_eval,
        compute_metrics = AMEMetric,
        callbacks=None if args.keep_all_ckpt else [RemoveBadSaveCallback()],
    )
    print("init trainer done, start training")
    trainer.train(resume_from_checkpoint=args.continue_from_last_ckpt)

if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
import datasets
import transformers
import argparse

torch.set_printoptions(precision=4, sci_mode=False, linewidth=180)

def parse():
    parser = argparse.ArgumentParser()

    # From here, path arguments
    parser.add_argument(
        "--pairs",
        type = str,
        default = "wecover/OPUS_Tatoeba",
        help = "translation pair path"
    )
    parser.add_argument(
        "--model",
        type = str,
        default = "intfloat/multilingual-e5-base",
        help = "Dataset_Sentence path"
    )
    parser.add_argument(
        "--size",
        type = int,
        default = 100,
        help = "size of similarity matrix"
    )
    parser.add_argument(
        "--direction",
        type = str,
        default = "both",
        choices = ["both", "left", "right", "cross"],
        help = "language of similarity map"
    )

    # device settings
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 64,
        help = "batch size"
    )
    parser.add_argument(
        "--device",
        type = int,
        default = -1,
        help = "gpu number, -1 means using cpu"
    )
    return parser.parse_args()

def encode(model, input):
    output = model(**input)
    token_embeddings = output["last_hidden_state"]
    input_mask_expanded = input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeds = F.normalize(embeds, p = 2, dim = 1)
    return output, embeds

def compute_sim_matrix(embed):
    embed_tensor = torch.cat(embed, dim = 0)
    return torch.mm(embed_tensor, embed_tensor.t())

def make_sim_matrix():
    with torch.no_grad():
        args = parse()
        model = transformers.AutoModel.from_pretrained(args.model, add_pooling_layer = False)
        model.eval()
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        except:
            tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
        data = datasets.load_dataset(args.pairs, data_files=args.pairs.split("_")[1]+"/en-ko/train.parquet", split = "train[:" + str(args.size) + "]")
        
        device = 'cpu' if args.device == -1 else "gpu:" + str(args.device)
        model = model.to(device)

        def tokenize_function_1(examples):
            if "e5" in args.model:
                examples["sentence1"] = ["query: " + text for text in examples["sentence1"]]
            sentence1_encoding = tokenizer(examples["sentence1"], padding="max_length", truncation=True, max_length = 512)
            return {
                "input_ids" : sentence1_encoding["input_ids"],
                "attention_mask" : sentence1_encoding["attention_mask"],
            }
        
        def tokenize_function_2(examples):
            if "e5" in args.model:
                examples["sentence2"] = ["query: " + text for text in examples["sentence2"]]
            sentence2_encoding = tokenizer(examples["sentence2"], padding="max_length", truncation=True, max_length = 512)
            return {
                "input_ids" : sentence2_encoding["input_ids"],
                "attention_mask" : sentence2_encoding["attention_mask"],
            }

        print("Running at", device)
        if args.direction == "left":
            print("Language :", data[0]["lang1"])
            tokenized_data_1 = data.map(tokenize_function_1, batched=True, num_proc = 8)

            embeds_1 = []
            for idx in range(0, args.size, args.batch_size):
                batch = tokenized_data_1[idx:idx+args.batch_size]
                data_tensor = {
                    "input_ids" : torch.tensor(batch["input_ids"]),
                    "attention_mask" : torch.tensor(batch["attention_mask"]),
                }
                embeds_1.append(encode(model, data_tensor)[1])
            sim = compute_sim_matrix(embeds_1)
        elif args.direction == "right":
            print("Language :", data[0]["lang2"])
            tokenized_data_2 = data.map(tokenize_function_2, batched=True, num_proc = 8)

            embeds_2 = []
            for idx in range(0, args.size, args.batch_size):
                batch = tokenized_data_2[idx:idx+args.batch_size]
                data_tensor = {
                    "input_ids" : torch.tensor(batch["input_ids"]),
                    "attention_mask" : torch.tensor(batch["attention_mask"]),
                }
                embeds_2.append(encode(model, data_tensor)[1])
            sim = compute_sim_matrix(embeds_2)
        elif args.direction == "both":
            print("Language :", data[0]["lang1"], ",", data[0]["lang2"])

            tokenized_data_1 = data.map(tokenize_function_1, batched=True, num_proc = 8)

            embeds_1 = []
            for idx in range(0, args.size, args.batch_size):
                batch = tokenized_data_1[idx:idx+args.batch_size]
                data_tensor = {
                    "input_ids" : torch.tensor(batch["input_ids"]),
                    "attention_mask" : torch.tensor(batch["attention_mask"]),
                }
                embeds_1.append(encode(model, data_tensor)[1])

            tokenized_data_2 = data.map(tokenize_function_2, batched=True, num_proc = 8)

            embeds_2 = []
            for idx in range(0, args.size, args.batch_size):
                batch = tokenized_data_2[idx:idx+args.batch_size]
                data_tensor = {
                    "input_ids" : torch.tensor(batch["input_ids"]),
                    "attention_mask" : torch.tensor(batch["attention_mask"]),
                }
                embeds_2.append(encode(model, data_tensor)[1])
            sim = (compute_sim_matrix(embeds_1) + compute_sim_matrix(embeds_2)) / 2
        else:
            print("Language : crosslingual, ", data[0]["lang1"], ",", data[0]["lang2"])

            tokenized_data_1 = data.map(tokenize_function_1, batched=True, num_proc = 8)

            embeds_1 = []
            for idx in range(0, args.size, args.batch_size):
                batch = tokenized_data_1[idx:idx+args.batch_size]
                data_tensor = {
                    "input_ids" : torch.tensor(batch["input_ids"]),
                    "attention_mask" : torch.tensor(batch["attention_mask"]),
                }
                embeds_1.append(encode(model, data_tensor)[1])

            tokenized_data_2 = data.map(tokenize_function_2, batched=True, num_proc = 8)

            embeds_2 = []
            for idx in range(0, args.size, args.batch_size):
                batch = tokenized_data_2[idx:idx+args.batch_size]
                data_tensor = {
                    "input_ids" : torch.tensor(batch["input_ids"]),
                    "attention_mask" : torch.tensor(batch["attention_mask"]),
                }
                embeds_2.append(encode(model, data_tensor)[1])
            embed_tensor_1 = torch.cat(embeds_1, dim = 0)
            embed_tensor_2 = torch.cat(embeds_2, dim = 0)
            sim = torch.mm(embed_tensor_1, embed_tensor_2.t())
        print(sim)
if __name__ == "__main__":
    make_sim_matrix()
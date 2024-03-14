from transformers import AutoModel
import argparse

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type = str,
        required = True
    )
    parser.add_argument(
        "--model_path",
        type = str,
        required = True
    )
    return parser.parse_args()

def main():
    args = parse()
    model = AutoModel.from_pretrained(args.model_path)
    model.push_to_hub(args.model_name, private = True)

main()
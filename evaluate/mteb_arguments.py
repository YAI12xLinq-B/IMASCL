import argparse
def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type = str,
        required = True
    )
    parser.add_argument(
        "--evaluation_task",
        type = str,
        required = False
    )
    parser.add_argument(
        "--prompt",
        type = str,
        required = False
    )
    parser.add_argument(
        "--langs",
        nargs='+', 
        default=[], 
        help='choose corresponding language to evaluate'
    )
    parser.add_argument(
        "--tokenizer",
        type = str,
        required = True
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 256
    )
    parser.add_argument(
        '--task-types', 
        nargs='+', 
        default=[], 
        help='task types to evaluate'
    )
    return parser.parse_args()
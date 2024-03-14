import json
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type = str,
        default= "./output/tmp.json"
    )
    parser.add_argument(
        "--langs",
        nargs = "+",
        default = ["en", "ko", "fr", "ja", "ru"]
    )
    parser.add_argument(
        "--mono2mult",
        action="store_true"
    )
    return parser.parse_args()

args = parse()
args.langs = sorted(args.langs)

losses = []
tatoebas = []

for lang1 in args.langs:
    for lang2 in args.langs:
        if lang1 < lang2:
            if args.mono2mult:
                if not(lang1 == "en" or lang2 == "en"):
                    continue
            losses.append("eval_"+lang1+'.'+lang2+"_loss")
            tatoebas.append("eval_"+lang1+'.'+lang2+"_tatoeba_acc_avg")

loss_dict = {}

with open(args.config_path, 'r') as file:
    data = json.load(file)
    log_history = data["log_history"]
    for log in log_history:
        for idx, loss in enumerate(losses):
            if loss in log:
                if log["step"] in loss_dict:
                    loss_dict[log["step"]][0] += log[loss]
                    loss_dict[log["step"]][2] += log[tatoebas[idx]]
                    loss_dict[log["step"]][1] += 1
                else:
                    loss_dict[log["step"]] = [log[loss], 1, log[tatoebas[idx]]]

min_loss = 999999999
max_tatoeba = -1
min_step = -1
max_tatoeba_step = -1

for key, value in loss_dict.items():
    if min_loss >= (value[0] / value[1]):
        min_loss = (value[0] / value[1])
        min_step = key
    if max_tatoeba <= (value[2] / value[1]):
        max_tatoeba = (value[2] / value[1])
        max_tatoeba_step = key
print("loss")
print(min_loss)
print(min_step)
print("tatoeba")
print(max_tatoeba)
print(max_tatoeba_step)
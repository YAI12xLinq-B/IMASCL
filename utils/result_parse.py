import json
import argparse
import os

task2metric = {
    "Tatoeba":[("en_ko_accuracy", "[\"test\"][\"kor-eng\"][\"accuracy\"]"),
               ("en_ko_f1", "[\"test\"][\"kor-eng\"][\"f1\"]"),
               ("en_fr_accuracy", "[\"test\"][\"fra-eng\"][\"accuracy\"]"),
               ("en_fr_f1", "[\"test\"][\"fra-eng\"][\"f1\"]"),
               ("en_ja_accuracy", "[\"test\"][\"jpn-eng\"][\"accuracy\"]"),
               ("en_ja_f1", "[\"test\"][\"jpn-eng\"][\"f1\"]"),
               ("en_ru_accuracy", "[\"test\"][\"rus-eng\"][\"accuracy\"]"),
               ("en_ru_f1", "[\"test\"][\"rus-eng\"][\"f1\"]")],
    "Tatoeba_AVG":[("@Tatoeba@accuracy", ["en_ko_accuracy", "en_fr_accuracy", "en_ja_accuracy", "en_ru_accuracy"])],
    "BUCC" : [("en_fr_accuracy", "[\"test\"][\"fr-en\"][\"accuracy\"]"),
              ("en_fr_f1", "[\"test\"][\"fr-en\"][\"f1\"]"),
              ("en_ru_accuracy", "[\"test\"][\"ru-en\"][\"accuracy\"]"),
              ("en_ru_f1", "[\"test\"][\"ru-en\"][\"f1\"]")],
    "BUCC_AVG": [("@BUCC@accuracy", ["en_fr_accuracy", "en_ru_accuracy"])],
    "STS12": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
    "STS13": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
    "STS14": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
    "STS15": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
    "STS16": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
    "STS17": [("en_cos_sim_spearman", "[\"test\"][\"en-en\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"en-en\"][\"cos_sim\"][\"pearson\"]"),
              ("ko_cos_sim_spearman", "[\"test\"][\"ko-ko\"][\"cos_sim\"][\"spearman\"]"),
              ("ko_cos_sim_pearson", "[\"test\"][\"ko-ko\"][\"cos_sim\"][\"pearson\"]"),
              ("en_fr_cos_sim_spearman", "[\"test\"][\"fr-en\"][\"cos_sim\"][\"spearman\"]"),
              ("en_fr_cos_sim_pearson", "[\"test\"][\"fr-en\"][\"cos_sim\"][\"pearson\"]")],
    "STS22": [("en_cos_sim_spearman", "[\"test\"][\"en\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"en\"][\"cos_sim\"][\"pearson\"]"),
              ("fr_cos_sim_spearman", "[\"test\"][\"fr\"][\"cos_sim\"][\"spearman\"]"),
              ("fr_cos_sim_pearson", "[\"test\"][\"fr\"][\"cos_sim\"][\"pearson\"]"),
              ("ru_cos_sim_spearman", "[\"test\"][\"ru\"][\"cos_sim\"][\"spearman\"]"),
              ("ru_cos_sim_pearson", "[\"test\"][\"ru\"][\"cos_sim\"][\"pearson\"]")],
    "STSBenchmark": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
                     ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
    "STS_AVG": [("en_cos_sim_pearson", ["STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STS22", "STSBenchmark"]),
                ("en_cos_sim_spearman", ["STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STS22", "STSBenchmark"]),
                ("ko_cos_sim_pearson", ["STS17"]),
                ("ko_cos_sim_spearman", ["STS17"]),
                ("fr_cos_sim_pearson", ["STS22"]),
                ("fr_cos_sim_spearman", ["STS22"]),
                ("ru_cos_sim_pearson", ["STS22"]),
                ("ru_cos_sim_spearman", ["STS22"]),
                ("en_fr_cos_sim_pearson", ["STS17"]),
                ("en_fr_cos_sim_spearman", ["STS17"])],
    "BIOSSES": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
    "SICK-R": [("en_cos_sim_spearman", "[\"test\"][\"cos_sim\"][\"spearman\"]"),
              ("en_cos_sim_pearson", "[\"test\"][\"cos_sim\"][\"pearson\"]")],
}

def load_score(model, task, metric, score, args):
    if type(score) == str:
        path = args.dir + model + "/" + task + ".json"
        if os.path.exists(path):
            with open(path, 'r') as file:
                json_data = json.load(file)
                try:
                    out = eval("json_data"+score)
                except:
                    out = "-"
        else:
            out = "-"
        return out
    else:
        sum_score = 0
        cnt = 0
        if metric[0] == '@':
            tmp = metric[1:].split("@")
            for sub_metric in score:
                for tmp_metric, tmp_score in task2metric[tmp[0]]:
                    if tmp_metric == sub_metric:
                        tmp_out = load_score(model, tmp[0], tmp_metric, tmp_score, args)
                        if tmp_out != "-":
                            sum_score += tmp_out
                            cnt += 1
        else:
            for sub_task in score:
                for tmp_metric, tmp_score in task2metric[sub_task]:
                    if tmp_metric == metric:
                        tmp_out = load_score(model, sub_task, metric, tmp_score, args)
                        if tmp_out != "-":
                            sum_score += tmp_out
                            cnt += 1
        if cnt == 0:
            return "-"
        else:
            return sum_score / cnt

def read_result(args):
    print("idx, model", end = "")
    for task in args.tasks:
        metrics = task2metric[task]
        for metric, _ in metrics:
            if metric[0] == "@":
                tmp = metric[1:].split("@")
                print(", " + task + "_" + tmp[1], end = "")
            else:
                print(", " + task + "_" + metric, end = "")
    
    print()
    
    idx = 0
    for model in args.models:
        print(str(idx).zfill(3) + ", " + model, end = "")
        for task in args.tasks:
            metrics = task2metric[task]
            for metric, score in metrics:
                out = load_score(model, task, metric, score, args)
                if type(out) != str:
                    out = str(round(out, 3))
                print(", " + out, end = "")
        idx += 1
        print()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type = str,
        default= "./results/"
    )
    parser.add_argument(
        "--models",
        nargs="+"
    )
    parser.add_argument(
        "--tasks",
        nargs="+"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    read_result(args)
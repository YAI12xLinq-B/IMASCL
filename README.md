<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=venom&height=200&text=Improving%20Multi-lingual%20Alignment-nl-Through-nl-Soft%20Contrastive%20Learning&fontSize=50&color=0:8871e5,100:b678c4&fontColor=AC96FB" alt="header"/>
</a></p>

<p align="center">
    <a href="">
        <img alt="GitHub release" src="https://img.shields.io/badge/arXiv-comming soon-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/datasets/wecover/OPUS">
        <img alt="Dataset release" src="https://img.shields.io/badge/Dataset-1.svg">
    </a>
    <a href="https://huggingface.co/wecover/ours_me5_baseline_me5_gpu2_en_ru_ja_fr_ko_priority_en">
        <img alt="Model release" src="https://img.shields.io/badge/Model-1.svg">
    </a>
</p>

## Install Dependencies
```
pip install -r requirements.txt
```

## Train
Just run the code below to reproduce our multilingual result.
```
./train.sh
```

If you want to modify the settings or parameter, fix arguments in [train.sh]().

Here are some important arguments.

- model : Student model path.
- embed_model : Teacher model path.
- continue_from_last_ckpt : Continue from the last ckpt in the output dir
- dataset_sentence : Train dataset path
- dataset_eval : Validation dataset path
- langs : Languages to train. find the language codes at [here](https://huggingface.co/datasets/wecover/OPUS)

See [train.py]() to find more detailed information for each arguments.

## Evaluate
Run the code below to evaluate model on Bitext Mining and STS.
```
python ./evaluate/run_mteb.py --model_name model_path(or local) --langs en ko fr ru ja eng kor fra rus jpn --tokenizer xlm-roberta-base --batch_size 512 --task-types BitextMining STS
```

Here are informations of each argument.

- model_name : Path of model to evaluate
- langs : Languages to evaluate, find the language codes at [here](https://huggingface.co/datasets/wecover/OPUS) and [Tatoeba(MTEB)](https://huggingface.co/datasets/mteb/tatoeba-bitext-mining/blob/main/README.md)
- tokenizer : Path of tokenizer. For our models, use same tokenizer with the student model.

You can find the evaluation results on ./results/

## Parse Evaluate Results
To parse the result to CSV file, run the code below.

```
python ./evaluate/result_parse.py --tasks Tatoeba_AVG BUCC_AVG STS_AVG \
--models model_paths(e.g. wecover/ours/)  > ./table.csv
```

To parse all evaluation results, run the code below.

```
python ./evaluate/result_parse.py --tasks Tatoeba_AVG Tatoeba BUCC_AVG BUCC STS_AVG STS \
--models model_paths(e.g. wecover/ours/) > ./table.csv
```

<br>
<br>
<br>
Please leave me any issues.
import torch
from transformers import TrainerCallback
import os
import shutil

def AMEMetric(pred):
    with torch.no_grad():
        output = pred.predictions
        embeds = torch.tensor(output)
        embeds_1 = embeds[:, 0, :]
        embeds_2 = embeds[:, 1, :]
        sim_matrix = torch.cdist(embeds_1, embeds_2)
        _, maxidx_1_to_2 = torch.min(sim_matrix, dim=1)
        _, maxidx_2_to_1 = torch.min(sim_matrix, dim=0)
        mat_size = sim_matrix.size(0)
        tatoeba_acc_1_to_2 = torch.sum(maxidx_1_to_2 == torch.arange(mat_size)) / mat_size
        tatoeba_acc_2_to_1 = torch.sum(maxidx_2_to_1 == torch.arange(mat_size)) / mat_size
        
        return {
            "tatoeba_acc_1_to_2" : tatoeba_acc_1_to_2,
            "tatoeba_acc_2_to_1" : tatoeba_acc_2_to_1,
            "tatoeba_acc_avg" : (tatoeba_acc_1_to_2 + tatoeba_acc_2_to_1) / 2
        }

class RemoveBadSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        
        best_model_steps = {}
        best_model_metric = {}
        metric_for_save_model_langs = []
        larger_better_dict = {}
        
        for logs in state.log_history:
            for idx, metric_portion in enumerate(args.metrics_for_save_model):
                for key in logs.keys():
                    if (metric_portion in key) and ("per_second" not in key) and key not in metric_for_save_model_langs:
                        metric_for_save_model_langs.append(key)
                        best_model_steps[key] = -1
                        best_model_metric[key] = 0 if args.metrics_larger_better[idx] else 9999999
                        larger_better_dict[key] = args.metrics_larger_better[idx]
            
            for metric in metric_for_save_model_langs:
                if metric in logs:
                    if larger_better_dict[metric]:
                        if logs[metric] >= best_model_metric[metric]:
                            best_model_metric[metric] = logs[metric]
                            best_model_steps[metric] = logs["step"]
                    else:
                        if logs[metric] <= best_model_metric[metric]:
                            best_model_metric[metric] = logs[metric]
                            best_model_steps[metric] = logs["step"]

        print("best model steps: " + str(best_model_steps))
        for dirname in os.listdir(args.output_dir):
            if len(dirname.split("-")) == 2:
                if int(dirname.split("-")[-1]) not in best_model_steps.values():
                    if args.local_rank % args.world_size == 0:
                        shutil.rmtree(os.path.join(args.output_dir, dirname))
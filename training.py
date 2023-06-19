import math
import torch

from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer_utils import set_seed
from torch.utils.data import DataLoader


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)

'''
    Need to override the trainer to provide different custom dataloaders for training and evaluation.
    Since we are performing dynamic NSP, we need to provide the dataset text during dataloader processing.
    This is efficiently handled since the dataset is memory-mapped and there is very less actual transfer of data.
'''
class PretrainingTrainer(Trainer):
    def __init__(self, doc_lens=None, **kwargs):
        super().__init__(**kwargs)
        self.doc_lens = doc_lens

    def get_train_dataloader(self):
        train_collator = DataCollatorForPretraining(tokenizer=self.tokenizer, dataset=self.train_dataset, doc_lens=self.doc_lens, sample_pairs=True)
        return DataLoader(self.train_dataset, 
                batch_size=self.args.train_batch_size, 
                sampler=self._get_train_sampler(), 
                collate_fn=train_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
                prefetch_factor=32)

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_collator = DataCollatorForPretraining(tokenizer=self.tokenizer, dataset=eval_dataset, doc_lens=self.doc_lens, sample_pairs=True)
        return DataLoader(eval_dataset, 
                batch_size=self.args.eval_batch_size, 
                sampler=self._get_eval_sampler(eval_dataset), 
                collate_fn=eval_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                prefetch_factor=32)
'''
    Called after every forward pass
    logits here is a tuple containing all elements of ModelOutput, in order, except loss
    labels is a tuple of all elements having suffix "labels" in BatchEncoding 
'''
def preprocess_logits_for_metrics(logits, labels):
    mlm_loss = logits[1]
    nsp_preds = logits[0].argmax(dim=-1)
    return (nsp_preds, mlm_loss.unsqueeze(-1))

'''
    Called after every epoch
    p is a named tuple having two keys
    "predictions" is the concatenation (across all forward passes) of tuple returned by the above function
    "labels" is the concatenation (across all forward passes) of "labels" input in the above function
'''
def compute_metrics(p):
    mlm_loss = p.predictions[1]
    metrics = {}
    metrics['perplexity'] = math.exp(mlm_loss.mean())
    metrics['sr_f1'] = f1_score(p.predictions[0], p.label_ids[1], average='macro')
    return metrics





    




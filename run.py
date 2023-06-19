import json
import os
import sys

from collections import defaultdict
from datasets import load_dataset, Features, Sequence
from datasets.features import Value
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, AutoModel

from data_helpers import *
from model import *
from training import *

###################################################
SOURCE_PATH = "nlpaueb/legal-bert-base-uncased"
OUTPUT_PATH = "InLegalBERT"
CACHE_PATH = "Cache"
FROM_SCRATCH = False
###################################################


'''
    We use HuggingFace datasets library since this provides memory-mapped datasets.
    This means that the entire data is maintained as a table of addresses in the main memory.
    True data is fetched from the secondary memory only when needed.
    This enables uniform implementation even for very large datasets.
    This also enables easily passing the datasets around different components of the pipeline. 
'''
if __name__ == '__main__':
    __spec__ = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1.'
    sys.setrecursionlimit(25000)

    print("*** Loading Dataset ***")
    dataset = load_dataset('json', data_files={'train': "train.jsonl", 'test': "test.jsonl"}, cache_dir=CACHE_PATH)

    schema = Features({'id': Value('string'), 'source': Value('string'), 'title': Value('string'), 'text': Sequence(Value('string'))})
    dataset = dataset.map(schema.encode_example, features=schema)

    config = AutoConfig.from_pretrained(SOURCE_PATH, cache_dir=CACHE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_PATH, cache_dir=CACHE_PATH)
    
    print("*** Tokenizing Dataset ***")
    tokenized_dataset = dataset.map(lambda e: tokenizer(list(e['text']), add_special_tokens=False, return_attention_mask=False), batched=False, num_proc=8)

    # Choose between chunk_text_blocks and chunk_text_line_by_line here
    print("*** Chunking into blocks/sentences ***")
    doc_lens = defaultdict(int)
    
    tokenized_dataset['train'] = tokenized_dataset['train'].map(chunk_text_blocks, batched=True, batch_size=64, remove_columns=tokenized_dataset['train'].column_names, num_proc=8)
    
    tokenized_dataset['test'] = tokenized_dataset['test'].map(chunk_text_blocks, batched=True, batch_size=64, remove_columns=tokenized_dataset['test'].column_names, num_proc=8)
    
    tokenized_dataset.save_to_disk(CACHE_PATH)

    print("*** Calculating Document Lengths ***")
    if os.path.exists(os.path.join("Output", OUTPUT_PATH, "doc_lens.json")):
        with open(os.path.join("Output", OUTPUT_PATH, "doc_lens.json")) as fr:
            doc_lens = json.load(fr)
    else:
        doc_lens = defaultdict(int)
        for exp in tqdm(tokenized_dataset['train']):
            doc_lens[exp['sample_ids'].split(':')[0]] += 1
        for exp in tqdm(tokenized_dataset['test']):
            doc_lens[exp['sample_ids'].split(':')[0]] += 1

        with open(os.path.join("Output", OUTPUT_PATH, "doc_lens.json"), 'w') as fw:
            json.dump(doc_lens, fw, indent=4)
    

    print("*** Formatting to PyTorch ***")
    tokenized_dataset.set_format(type='torch', columns=['anchor_input_ids', 'pair_input_ids', 'sr_labels'], output_all_columns=True)
    
    tokenized_dataset.save_to_disk(CACHE_PATH)

    
    model = InLegalBertForPreTraining(config)
    
    if not FROM_SCRATCH:
        # First try to get the BERT model with all its heads
        try: 
            trained_bert_model = BertForPreTraining.from_pretrained(SOURCE_PATH, cache_dir=CACHE_PATH)
            model.load_state_dict(trained_bert_model.state_dict())
        # Some models do not have the pre-trained heads, so get only the BERT module
        except:
            trained_bert_model = AutoModel.from_pretrained(SOURCE_PATH, cache_dir=CACHE_PATH)
            model.bert.load_state_dict(trained_bert_model.state_dict())

    training_args = TrainingArguments(
        output_dir=os.path.join("Output", OUTPUT_PATH),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy='no',
        eval_steps=250,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,
        eval_accumulation_steps=8,
        num_train_epochs=4,
        max_steps=300000,
        logging_strategy='steps',
        logging_steps=20000,
        logging_first_step=True,
        save_strategy='steps',
        save_steps=20000,
        save_total_limit=5,
        seed=42,
        fp16=True,
        dataloader_num_workers=30,
        load_best_model_at_end=False,
        metric_for_best_model='perplexity',
        greater_is_better=False,
        group_by_length=False,
        dataloader_pin_memory=True,
        resume_from_checkpoint=True,
        gradient_checkpointing=False,
        label_names=['mlm_labels', 'sr_labels'],
        no_cuda=False,
        ignore_data_skip=True
    )

    trainer = PretrainingTrainer(
        model=model,
        args=training_args,
        doc_lens=doc_lens,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    
    if training_args.do_train:
        _, train_loss, train_metrics = trainer.train(resume_from_checkpoint=False)
        train_metrics['train_samples'] = len(tokenized_dataset['train'])
        config.save_pretrained(os.path.join("Output", OUTPUT_PATH))
        trainer.save_model()
        trainer.save_metrics('train', train_metrics)


    if training_args.do_eval:
        test_metrics = trainer.evaluate()
        test_metrics['eval_samples'] = len(tokenized_dataset['test'])
        trainer.save_metrics('eval', test_metrics)

        print("*** Results ***")
        for k, v in test_metrics.items():
            print(f'{k}: {v}')














    




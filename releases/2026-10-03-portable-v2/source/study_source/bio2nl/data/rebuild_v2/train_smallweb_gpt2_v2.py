"""Train the English-only SmallWeb GPT-2 from random weights on v2 data.

This entrypoint only runs when explicitly called after the integrated data gate.
The pinned GPT-2 repository supplies tokenizer/model shape, never pretrained weights.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer,GPT2Config,GPT2LMHeadModel,Trainer,TrainingArguments
from train_from_scratch_v2 import BinBlocks,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-unaccepted-staging-data',action='store_true');a=p.parse_args();r=a.root
 gate=r/'validation/integrated_release_acceptance.json'
 if not gate.exists() or (not json.loads(gate.read_text()).get('training_gate_pass') and not a.allow_unaccepted_staging_data):raise SystemExit('Data release has not passed integrated acceptance; training blocked.')
 tp=r/'tokenizers/gpt2_pinned/metadata.json';tm=json.loads(tp.read_text());tokenizer=AutoTokenizer.from_pretrained(r/'tokenizers/gpt2_pinned',local_files_only=True)
 torch.manual_seed(a.seed);cfg=GPT2Config(vocab_size=tokenizer.vocab_size,n_positions=1024,n_ctx=1024,n_embd=768,n_layer=12,n_head=12,bos_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
 model=GPT2LMHeadModel(cfg)
 train=r/'tokenized/gpt2_smallweb/train.bin';meta=json.loads(train.with_suffix('.json').read_text())
 if sha(train)!=meta['sha256']:raise SystemExit('Training bin hash mismatch')
 if a.output.exists():raise SystemExit(f'Refusing to overwrite {a.output}')
 args=TrainingArguments(output_dir=str(a.output),overwrite_output_dir=False,num_train_epochs=3,per_device_train_batch_size=16,gradient_accumulation_steps=4,
    learning_rate=3e-4,lr_scheduler_type='cosine',warmup_ratio=.02,weight_decay=.01,optim='adamw_torch',seed=a.seed,logging_steps=200,save_strategy='no',report_to=[],fp16=True,dataloader_num_workers=2)
 trainer=Trainer(model=model,args=args,train_dataset=BinBlocks(train,512));trainer.train();trainer.save_model(str(a.output));tokenizer.save_pretrained(a.output)
 (a.output/'data_provenance.json').write_text(json.dumps({'release_id':'2026-09-25-v2','input':str(train.relative_to(r)),'input_sha256':sha(train),'tokenizer_revision':tm['revision'],
    'tokenizer_sha256':tm['tokenizer_files_sha256'],'pretraining_seed':a.seed,'epochs':3,'tokens_per_epoch':meta['tokens'],'train_steps':trainer.state.global_step},indent=2)+'\n')

if __name__=='__main__':main()

"""Train one random-initialized matrix model from an accepted v2 token file.

This is an execution entrypoint, not part of dataset construction. It must be
called explicitly after the integrated acceptance gate passes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

SIZES={"tiny":{"n_layer":4,"n_head":4,"n_embd":256,"context":512},
       "small":{"n_layer":12,"n_head":12,"n_embd":768,"context":512}}

class BinBlocks(Dataset):
    def __init__(self,path,context):self.data=np.memmap(path,dtype="<u2",mode="r");self.context=context;self.n=len(self.data)//context
    def __len__(self):return self.n
    def __getitem__(self,index):
        x=torch.from_numpy(self.data[index*self.context:(index+1)*self.context].astype(np.int64,copy=True))
        return {"input_ids":x,"labels":x.clone()}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--condition',choices=('english','protein','shuffled','randomaa','dna'),required=True);p.add_argument('--tokenizer',choices=('pure_byte','mixed_bpe'),required=True);p.add_argument('--size',choices=SIZES,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=float,default=1);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--grad-accum',type=int,default=2);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--allow-unaccepted-staging-data',action='store_true');a=p.parse_args();root=a.root
    gate=root/'validation/integrated_release_acceptance.json'
    if not gate.exists() or (not json.loads(gate.read_text()).get('training_gate_pass') and not a.allow_unaccepted_staging_data):
        raise SystemExit('Data release has not passed integrated acceptance; training blocked.')
    token_path=root/'tokenized'/a.tokenizer/a.condition/'train.bin';meta=json.loads(token_path.with_suffix('.json').read_text())
    tokenizer_meta=json.loads((root/'tokenizers'/a.tokenizer/'metadata.json').read_text())
    assert sha(token_path)==meta['sha256'] and sha(root/'tokenizers'/a.tokenizer/'tokenizer.json')==tokenizer_meta['tokenizer_sha256']
    if a.tokenizer=='pure_byte':vocab_size=260
    else:vocab_size=json.loads((root/'tokenizers/mixed_bpe/tokenizer.json').read_text())['model']['vocab'].__len__()
    special=tokenizer_meta['special_tokens'];size=SIZES[a.size];torch.manual_seed(a.seed)
    cfg=GPT2Config(vocab_size=vocab_size,n_positions=size['context'],n_ctx=size['context'],n_embd=size['n_embd'],n_layer=size['n_layer'],n_head=size['n_head'],
        bos_token_id=special['<|eos_v2|>'],eos_token_id=special['<|eos_v2|>'],pad_token_id=special['<|pad_v2|>'])
    model=GPT2LMHeadModel(cfg)
    if a.output.exists():raise SystemExit(f'Output already exists; refusing to overwrite: {a.output}')
    args=TrainingArguments(output_dir=str(a.output),overwrite_output_dir=False,num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.learning_rate,lr_scheduler_type='cosine',warmup_ratio=.02,weight_decay=.01,
        optim='adamw_torch',seed=a.seed,logging_steps=200,save_strategy='no',report_to=[],fp16=True,dataloader_num_workers=2)
    trainer=Trainer(model=model,args=args,train_dataset=BinBlocks(token_path,size['context']))
    trainer.train();trainer.save_model(str(a.output))
    tokenizer_json=root/'tokenizers'/a.tokenizer/'tokenizer.json'
    (a.output/'tokenizer.json').write_bytes(tokenizer_json.read_bytes())
    (a.output/'data_provenance.json').write_text(json.dumps({'release_id':'2026-09-25-v2','data_file':str(token_path.relative_to(root)),
        'data_sha256':sha(token_path),'tokenizer':a.tokenizer,'tokenizer_sha256':sha(tokenizer_json),'condition':a.condition,'size':a.size,
        'pretraining_seed':a.seed,'tokens':meta['tokens'],'train_steps':trainer.state.global_step,'training_loss':trainer.state.log_history[-1].get('train_loss')},indent=2)+'\n')

if __name__=='__main__':main()

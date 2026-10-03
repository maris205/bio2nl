"""Fresh CPU input/invariant checks plus explicitly synthetic tiny-model checks."""
from __future__ import annotations
import argparse
import copy
import os
import numpy as np
import torch
from support import require,write,now

def numerical_checks(runtime,models):
    """No real examples or study weights; no GPU use and no training acceptance."""
    torch.set_num_threads(1)
    tiny={'pretraining':{'architecture':{'model_type':'gpt2','n_layer':1,'n_head':2,'n_embd':16,'n_inner':32,'activation_function':'gelu_new','attn_pdrop':0.,'embd_pdrop':0.,'resid_pdrop':0.,'layer_norm_epsilon':1e-5,'initializer_range':.02,'tie_word_embeddings':True},'context_length':16},'tokenizer':{'vocab_size':32,'eos_token_id':1,'pad_token_id':0}}
    a=models.create_pretraining_model(tiny,0);b=models.create_pretraining_model(tiny,0);c=models.create_pretraining_model(tiny,1)
    require(models.state_digest(a.state_dict())==models.state_digest(b.state_dict()) and models.state_digest(a.state_dict())!=models.state_digest(c.state_dict()),'Paired initialization test failed')
    classifier=models.create_classifier(a,1).eval();x=torch.tensor([[4,5,2,6,1,0,0,0],[7,2,8,9,1,0,0,0]]);mask=(x!=0).long()
    with torch.no_grad():
        logits=classifier(x,mask);short=classifier(x[:,:5],mask[:,:5]);hidden=classifier.backbone(input_ids=x,attention_mask=mask,return_dict=True).last_hidden_state
        manual=classifier.score(hidden[:,4]);require(torch.allclose(logits,short,rtol=0,atol=1e-6) and torch.equal(logits,manual),'Mask-based pooling/batch width test failed')
        lm=b.eval()(input_ids=x[:,:5],return_dict=True).logits[:,:-1].float();labels=x[:,1:5]
        channel=torch.nn.functional.cross_entropy(lm.transpose(1,2),labels,reduction='none')
        direct=-torch.log_softmax(lm,-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
        require(torch.allclose(channel,direct,atol=1e-6,rtol=0),'Corrected channel CE test failed')
    for mode in ('pretrain','sft'):
        _,names=runtime.optimizer_groups(classifier,mode,.01);flat=names['decay']+names['no_decay'];require(len(flat)==len(set(flat))==len(list(classifier.parameters())),'Optimizer parameter coverage failed')
    before=models.state_digest(classifier.state_dict());optimizer=torch.optim.AdamW(classifier.parameters(),lr=1e-3);loss=torch.nn.functional.cross_entropy(classifier(x,mask),torch.tensor([0,1]));loss.backward();runtime.checked_gradient_norm(classifier,1.);optimizer.step();require(models.state_digest(classifier.state_dict())!=before,'Tiny synthetic parameter update failed')
    budget=runtime.sft_budget(8044,'full');require(budget=={'epochs':5,'steps_per_epoch':252,'updates':1260,'sample_presentations':40220},'Full SFT budget changed')
    require(runtime.cosine_multiplier(0)==0 and runtime.cosine_multiplier(21)==1 and runtime.linear_multiplier(1260,1260)==0,'Schedule endpoints changed')
    return 8

def execute(protocol,sha256):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only explicit environment required')
    import runtime
    import model
    from runtime_data import ReleaseData
    data=ReleaseData(protocol,sha256,'prepare');runtime.validate_numerical_design(data.design,data.source_counts)
    checks=numerical_checks(runtime,model)
    for split in ('train','validation'):
        arrays,rows=data.load_source(split,smoke=False);require(len(rows)==data.source_counts[split],'Full source count mismatch');require(len(set(arrays['labels']))==2,'Source classes missing');checks+=1
    for condition,seed in [('EP',0),('ES',1),('EE',2)]:data.load_pretrain(condition,seed);checks+=1
    value={'schema_version':1,'status':'passed','at_utc':now(),'protocol_sha256':sha256,'prepared_manifest_sha256':data.manifest_sha256,'source_counts':data.source_counts,'checks':checks,'fixture_only_model_checks':True,'gpu_execution':False,'source_training_updates':0,'synthetic_tiny_fixture_updates':1,'training_acceptance_granted':False,'target_examples_read':0,'source_test_examples_read':0,'inputs':data.verify_current_inputs()}
    write(data.protocol_path.parent/'verification/cpu_checks.json',value);return value
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args();execute(a.protocol,a.protocol_sha256)

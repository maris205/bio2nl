"""Build paired, template-disjoint long-range subject-verb tasks.

For every fixed prefix/template, emit exactly one grammatical and one
ungrammatical sentence that differ only in the final verb number. Whole template
pairs stay within a single split. Distractor number is independently balanced,
so matching only the nearest noun does not solve the task.
"""
import argparse,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path

SG=["scientist","engineer","author","teacher","doctor","artist","pilot","manager","student","farmer","lawyer","nurse","chef","worker","player"]
PL=[x+"s" for x in SG]
VERBS={"run":("runs","run"),"write":("writes","write"),"speak":("speaks","speak"),"work":("works","work"),"read":("reads","read"),"walk":("walks","walk"),"sing":("sings","sing"),"win":("wins","win"),"lead":("leads","lead")}
PREPS=["near","beside","behind","beyond","under","above","against"]
ADJS=["tall","quiet","famous","clever","young","serious","curious","brave"]
SPLITS={"train":8000,"validation":1000,"test":1000} # base templates; each emits 2 rows
SEED=20260925

def stable(v):return hashlib.sha256(v.encode()).hexdigest()
def build_one(root,n_distractors):
    assert n_distractors%4==0
    phrases=n_distractors//4; rng=random.Random(SEED+n_distractors)
    rows={s:[] for s in SPLITS}; seen=set(); nearest_counts={s:Counter() for s in SPLITS}
    # Fill exact cells: subject number x final distractor number, balanced within split.
    for split,n_templates in SPLITS.items():
        per_cell=n_templates//4
        for head_number in (0,1):
            for nearest_number in (0,1):
                for i in range(per_cell):
                    while True:
                        head=rng.choice(SG if head_number==0 else PL)
                        dnums=[rng.randrange(2) for _ in range(max(0,phrases-1))]+[nearest_number]
                        distractors=[]
                        for number in dnums:
                            noun=rng.choice(SG if number==0 else PL)
                            distractors.extend([rng.choice(PREPS),"the",rng.choice(ADJS),noun])
                        root_verb=sorted(VERBS)[i%len(VERBS)]
                        sentence_prefix=" ".join(["The",head,*distractors])
                        template=sentence_prefix+" "+root_verb
                        if template not in seen:seen.add(template);break
                    sg_verb,pl_verb=VERBS[root_verb]
                    good=sg_verb if head_number==0 else pl_verb
                    bad=pl_verb if head_number==0 else sg_verb
                    rid=stable(f"longrange-v2:{n_distractors}:{split}:{template}")
                    # Store only model input, label and traceable template metadata.
                    rows[split].append(dict(task=f"longrange_d{n_distractors}",split=split,row_id=rid,
                        template_group_id=rid,sentence=sentence_prefix+" "+good+" .",label=1,
                        dependency_distance_words=n_distractors+1,head_number="singular" if head_number==0 else "plural",
                        nearest_distractor_number="singular" if nearest_number==0 else "plural",verb_root=root_verb))
                    rows[split].append(dict(task=f"longrange_d{n_distractors}",split=split,row_id=stable(rid+":bad"),
                        template_group_id=rid,sentence=sentence_prefix+" "+bad+" .",label=0,
                        dependency_distance_words=n_distractors+1,head_number="singular" if head_number==0 else "plural",
                        nearest_distractor_number="singular" if nearest_number==0 else "plural",verb_root=root_verb))
                    nearest_counts[split][(head_number,nearest_number)]+=1
        random.Random(SEED+n_distractors+{"train":0,"validation":100,"test":200}[split]).shuffle(rows[split])
    report={"task":f"longrange_d{n_distractors}","seed":SEED+n_distractors,"n_distractor_tokens":n_distractors,
      "template_counts":SPLITS,"row_counts":{s:len(rows[s]) for s in SPLITS},"construction":"paired minimal contrasts; same prefix/template, only verb number differs; templates disjoint across splits",
      "label_counts":{s:dict(Counter(r['label'] for r in rows[s])) for s in SPLITS},
      "subject_nearest_number_template_cells":{s:{str(k):v for k,v in nearest_counts[s].items()} for s in SPLITS},
      "baselines_accuracy":{},"split_overlap":{"template_groups":0,"sentences":0}}
    for split in SPLITS:
        pair_groups=defaultdict(list)
        for r in rows[split]:pair_groups[r['template_group_id']].append(r)
        assert all(len(v)==2 and {x['label'] for x in v}=={0,1} for v in pair_groups.values())
        assert all(v[0]['sentence'].rsplit(' ',2)[0]==v[1]['sentence'].rsplit(' ',2)[0] for v in pair_groups.values())
        y=[];nearest_pred=[];head_pred=[];majority=[]
        for v in pair_groups.values():
            head=0 if v[0]['head_number']=='singular' else 1
            near=0 if v[0]['nearest_distractor_number']=='singular' else 1
            # Predict grammatical if observed verb agrees with nearest noun.
            for r in v:
                is_singular=r['sentence'].split()[-2] in {verbs[0] for verbs in VERBS.values()}
                verb_number=0 if is_singular else 1
                y.append(r['label']);nearest_pred.append(int(verb_number==near));head_pred.append(int(verb_number==head));majority.append(0)
        def acc(pred):return sum(x==z for x,z in zip(y,pred))/len(y)
        report['baselines_accuracy'][split]={"majority":acc(majority),"nearest_noun_agreement":acc(nearest_pred),"subject_verb_agreement_oracle":acc(head_pred)}
        assert report['baselines_accuracy'][split]['majority']==.5
        assert report['baselines_accuracy'][split]['nearest_noun_agreement']==.5
        assert report['baselines_accuracy'][split]['subject_verb_agreement_oracle']==1.0
    out=root/'data/synthetic'/f"longrange_d{n_distractors}";out.mkdir(parents=True,exist_ok=True)
    checks={}
    for split,values in rows.items():
        path=out/f"{split}.jsonl"
        with path.open('w') as f:
            for row in values:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        checks[split]={"file":str(path.relative_to(root)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"rows":len(values)}
    report['files']=checks;report['acceptance_pass']=True
    path=root/'validation'/f"synthetic_longrange_d{n_distractors}_acceptance.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    reports=[build_one(a.root,n) for n in (8,16,24)]
    manifest={'release_id':'2026-09-25-v2','seed':SEED,'tasks':[r['task'] for r in reports],
      'files':{r['task']:r['files'] for r in reports},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'notes':['The trainable input is sentence only; labels/metadata are separate.','All paired contrasts are assigned as one group and cannot cross split.','Nearest-noun agreement, majority and subject/verb baselines are reported.']}
    (a.root/'metadata/synthetic_longrange_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({r['task']:{'counts':r['row_counts'],'baselines':r['baselines_accuracy']} for r in reports},indent=2),flush=True)
if __name__=='__main__':main()

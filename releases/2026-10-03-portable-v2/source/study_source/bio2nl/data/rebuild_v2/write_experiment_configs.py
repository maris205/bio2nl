"""Write version-pinned input/config expectations; does not launch any model."""
import argparse
from pathlib import Path
from common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    matrix={"experiment":"matched random-initialized decoder-only pretraining",
      "conditions":{"bytelevel":["english","protein","shuffled","randomaa","dna"],
                    "mixed_bpe":["english","protein","shuffled","randomaa"]},
      "sizes":{"tiny":{"n_layer":4,"n_head":4,"n_embd":256,"context":512},
               "small":{"n_layer":12,"n_head":12,"n_embd":768,"context":512}},
      "initialization":"random GPT2Config; no pretrained weights",
      "per_condition_tokens":16_777_216,"training_blocks":32_768,"epochs":1,"pretraining_seeds":[0],
      "per_device_batch_size":16,"gradient_accumulation_steps":2,"learning_rate":0.0003,
      "scheduler":"cosine","warmup_ratio":0.02,"weight_decay":0.01,"optimizer":"adamw_torch",
      "historical_reference":"Current legacy run used pretrained seed 0 and three downstream fine-tune seeds. New training seeds remain an execution decision; report pretraining seeds separately from fine-tuning seeds.",
      "validation_bins":"Separate 262144-token validation bin for each condition and tokenizer; use only for competence and model selection within a frozen protocol.",
      "test_bins":"Separate 262144-token test bin for held-out language-modeling evaluation; do not tune on it.",
      "data_release_id":"2026-09-25-v2","training_not_started":True}
    smallweb={"experiment":"GPT2-style English-only SmallWeb from random initialization","model_id_for_architecture":"openai-community/gpt2",
      "tokenizer_revision":"607a30d783dfa663caf39e06633721c8d4cfcd7e","weights":"not loaded; random initialization",
      "vocab_size":50257,"n_layer":12,"n_head":12,"n_embd":768,"model_positions":1024,
      "input_block_length":512,"train_tokens_per_epoch":49_999_872,"epochs":3,
      "per_device_batch_size":16,"gradient_accumulation_steps":4,"learning_rate":0.0003,
      "scheduler":"cosine","warmup_ratio":0.02,"weight_decay":0.01,"optimizer":"adamw_torch",
      "data_release_id":"2026-09-25-v2","seed":0,"training_not_started":True,
      "legacy_caveat":"Old entrypoint advertised --tokens but did not parse it; actual corpus exposure cannot be inferred from that help string.",
      "new_data_serialization":"append one EOS after every source document; fixed separate validation/test bins."}
    write_json(a.root/"configs/training_matrix_v2.json",matrix)
    write_json(a.root/"configs/smallweb_gpt2_v2.json",smallweb)
    print("wrote configs; training_not_started=true")

if __name__=="__main__":main()

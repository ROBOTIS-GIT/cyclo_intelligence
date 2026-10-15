"""Small, reproducible image+language LoRA training; no robot commands.

A one-sample run is a pipeline/overfit test, not generalization evaluation.
Masks remain separate annotations; this script trains the approved decision JSON.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import time


if __package__:
    from .data_format import FORMAT, messages_for, read_samples
    from .batching import collate_batches, assistant_ce
else:
    from data_format import FORMAT, messages_for, read_samples
    from batching import collate_batches, assistant_ce


def batch_on_device(batch, device):
    """Copy tensors for this step without moving the cached BatchFeature in place."""
    return {key: value.to(device) for key, value in batch.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True, help='Export directory containing train.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3.5-4B')
    p.add_argument('--revision', default='main')
    p.add_argument('--skip-baseline', action='store_true', help='Skip pre-training replay when repeating a verified experiment')
    p.add_argument('--skip-replay', action='store_true', help='Skip training-image replay for UI jobs')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.output.exists():
        raise ValueError('Use positive steps and a new output directory')
    source = args.dataset / 'train.jsonl'
    samples = read_samples(source)
    if not samples:
        raise ValueError('Training split is empty; collect independent training scenes')
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(42)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
        min_pixels=256*32*32, max_pixels=768*32*32)
    processor.tokenizer.padding_side = "right"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, revision=args.revision,
        dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    targets = [name for name, module in model.named_modules()
               if 'language_model.layers.' in name and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError('No language backbone linear layers found')
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        bias='none', task_type='CAUSAL_LM', target_modules=targets))
    model.config.use_cache = False
    batches = []; prompts = []
    for sample in samples:
        messages = messages_for(sample, args.dataset)
        full = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=False, return_dict=True, return_tensors='pt')
        prefix = processor.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors='pt')
        count = prefix['input_ids'].shape[1]
        if full['input_ids'].shape[1] > 4096 or not torch.equal(full['input_ids'][:,:count], prefix['input_ids']):
            raise ValueError('Input too long or assistant prefix mismatch; refusing incorrect loss masking')
        labels = full['input_ids'].clone(); labels[:,:count] = -100
        if not torch.any(labels != -100):
            raise ValueError('No assistant target tokens')
        full['labels'] = labels
        batches.append(full); prompts.append(prefix)
        messages[1]['content'][0]['image'].close()
    def generate(prefix):
        count = prefix['input_ids'].shape[1]
        inp = {k:v.to(model.device) for k,v in prefix.items()}
        model.eval()
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=192, do_sample=False, use_cache=True)
        return processor.decode(out[0,count:], skip_special_tokens=True).strip()
    baseline = [] if args.skip_baseline else [generate(prefix) for prefix in prompts]
    print(json.dumps({'phase':'baseline','predictions':baseline},ensure_ascii=False),flush=True)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    losses = []; started = time.monotonic()
    order = list(range(len(batches))); rng = random.Random(42); cursor = 0
    for step in range(args.steps):
        selected=[]
        for _ in range(args.batch_size):
            if cursor % len(order) == 0:rng.shuffle(order)
            selected.append(batches[order[cursor % len(order)]]);cursor+=1
        optimizer.zero_grad(set_to_none=True)
        batch = batch_on_device(collate_batches(selected, processor.tokenizer.pad_token_id), model.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = assistant_ce(model,batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step+1) % 5 == 0:
            print(json.dumps({'step':step+1,'loss':losses[-1],'batch_size':args.batch_size,'samples_seen':cursor, 'elapsed_seconds':round(time.monotonic()-started,1),
                'gpu_allocated_mib':round(torch.cuda.memory_allocated()/1024**2,1),
                'gpu_reserved_mib':round(torch.cuda.memory_reserved()/1024**2,1)}),flush=True)
        del batch, loss
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output); processor.save_pretrained(args.output)
    predictions = [] if args.skip_replay else [generate(prefix) for prefix in prompts]
    report = {'mode':'training_only' if args.skip_replay else 'training_sample_replay_not_generalization', 'model':args.model,
        'model_input_format':FORMAT,
        'revision':getattr(model.config,'_commit_hash',args.revision), 'samples':len(samples),
        'shuffle_seed':42,'baseline_skipped':args.skip_baseline,'replay_skipped':args.skip_replay,
        'batch_size':args.batch_size,'gradient_accumulation':1,'samples_seen':cursor,
        'steps':args.steps,'learning_rate':args.learning_rate,'rank':8,'losses':losses,
        'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'baseline':baseline,'after_training':predictions,
        'targets':[json.loads(s['messages'][-1]['content'][0]['text']) for s in samples],
        'max_gpu_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
    (args.output/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='losses'},ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()

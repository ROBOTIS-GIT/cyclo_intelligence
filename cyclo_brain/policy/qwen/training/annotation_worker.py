"""Offline train/batch worker. Reads saved frames; never talks to robot services."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from qwen_annotation_contract import system_for, prompt_for


def emit(**event):
    print(json.dumps(event,ensure_ascii=False),flush=True)


def image_path(row):
    root=Path('/workspace/dataset/reasoning').resolve()
    path=(root/row['image']).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
        raise ValueError('Saved frame is missing or changed')
    return path


def train(job):
    with tempfile.TemporaryDirectory(prefix='qwen-annotation-') as temp:
        dataset=Path(temp)
        samples=[]
        for index,row in enumerate(job['frames']):
            name=f'frame_{index:04d}.jpg'
            shutil.copyfile(image_path(row),dataset/name)
            sample=row['sample']
            sample['messages'][1]['content'][0]['image']=name
            samples.append(json.dumps(sample,ensure_ascii=False))
        (dataset/'train.jsonl').write_text('\n'.join(samples)+'\n')
        command=[sys.executable,'-u',str(Path(__file__).with_name('train_lora.py')),
            '--dataset',str(dataset),'--output',job['adapter_path'],
            '--model',job['base_model'],'--revision',job['revision'],
            '--steps',str(job['steps']),'--batch-size',str(job.get('batch_size',4)),'--learning-rate','5e-5','--skip-baseline','--skip-replay']
        emit(event='progress',phase='Loading Qwen',step=0)
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            print(line.rstrip(),flush=True)
            try: value=json.loads(line)
            except ValueError: continue
            if 'step' in value: emit(event='progress',phase='Fine-tuning Qwen',**value)
        exit_code=process.wait()
        if exit_code:
            raise RuntimeError(f'Qwen fine-tuning failed (exit code {exit_code}); see worker.log for the original error')


def generate(job):
    os.environ.update(QWEN_MODEL_ID=job['base_model'],QWEN_MODEL_REVISION=job['revision'],
                      QWEN_ADAPTER_PATH=job['adapter_path'],QWEN_QUANTIZATION='4bit')
    from qwen_engine.engine import QwenEngine
    from PIL import Image
    engine=QwenEngine()
    emit(event='progress',phase='Loading tuned Qwen')
    engine.load()
    try:
        for row in job['frames']:
            try:
                with Image.open(image_path(row)) as original:
                    image=original.convert('RGB')
                try:
                    result=engine.generate(SimpleNamespace(system_prompt=system_for(row['frame'].get('context') or {}),
                        prompt=prompt_for(row['frame'],row['catalog']),
                        max_new_tokens=512,enable_thinking=False,do_sample=False),image)
                finally: image.close()
                if result.get('finish_reason')!='stop': raise ValueError('Output was truncated')
                emit(event='prediction',episode_id=row['episode_id'],frame_id=row['frame_id'],text=result['text'])
            except Exception as exc:
                emit(event='frame_error',episode_id=row['episode_id'],frame_id=row['frame_id'],error=str(exc))
    finally: engine.unload()


def main():
    job=json.loads(Path(sys.argv[1]).read_text())
    if job['kind'] in ('training','reasoning_training'): train(job)
    else: generate(job)


if __name__=='__main__': main()

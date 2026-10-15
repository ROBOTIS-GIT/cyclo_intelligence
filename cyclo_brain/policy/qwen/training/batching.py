"""Right-pad cached multimodal samples; supervise only assistant positions."""

def collate_batches(samples, pad_token_id):
    import torch
    if not samples:raise ValueError('Cannot collate an empty batch')
    keys=set(samples[0])
    if any(set(sample)!=keys for sample in samples):raise ValueError('Inconsistent processor tensor keys')
    length=max(sample['input_ids'].shape[1] for sample in samples)
    sequence={'input_ids','attention_mask','labels','token_type_ids','mm_token_type_ids'}
    result={}
    for key in keys:
        if key in sequence:
            fill=pad_token_id if key=='input_ids' else -100 if key=='labels' else 0
            values=[]
            for sample in samples:
                value=sample[key]
                if value.shape[0]!=1:raise ValueError('Expected individually cached samples')
                padded=value.new_full((1,length),fill);padded[:,:value.shape[1]]=value;values.append(padded)
            result[key]=torch.cat(values,dim=0)
        else:
            # Images stay in conversation order, including repeated samples.
            result[key]=torch.cat([sample[key] for sample in samples],dim=0)
    return result


def assistant_ce(model, batch):
    import torch.nn.functional as F
    base=model.get_base_model()
    inputs={key:value for key,value in batch.items() if key!='labels'}
    hidden=base.model(**inputs,use_cache=False).last_hidden_state
    rows,positions=(batch['labels'][:,1:]!=-100).nonzero(as_tuple=True)
    if not len(rows):raise ValueError('No assistant target tokens')
    logits=base.lm_head(hidden[rows,positions])
    return F.cross_entropy(logits.float(),batch['labels'][rows,positions+1])

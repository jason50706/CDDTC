import torch
from args import args, config
from tqdm import tqdm
from items_dataset import items_dataset

def test_predict(test_loader, device, model, min_label=1, max_label=3):
  model.eval()
  result = []
  
  for i, test_batch in enumerate(tqdm(test_loader)):
    batch_text = test_batch['batch_text']
    input_ids = test_batch['input_ids'].to(device)
    token_type_ids = test_batch['token_type_ids'].to(device)
    attention_mask = test_batch['attention_mask'].to(device)
    #labels = test_batch['labels'].to(device)
    crf_mask = test_batch["crf_mask"].to(device)
    sample_mapping = test_batch["overflow_to_sample_mapping"]
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=None, crf_mask=crf_mask)
    if args.use_crf:
      prediction = model.crf.decode(output[0], crf_mask)
    else:
      prediction = torch.max(output[0], -1).indices

    #make result of every sample
    sample_id = -1
    sample_result= {"text_a" : test_batch['batch_text'][0]}
    for batch_id in range(len(sample_mapping)):
        change_sample = False
        if sample_id != sample_mapping[batch_id]: change_sample = True
        #print(i, id)
        if change_sample:
          sample_id = sample_mapping[batch_id]
          sample_result= {"text_a" : test_batch['batch_text'][sample_id]}
          decode_span_table = torch.zeros(len(test_batch['batch_text'][sample_id]))

        spans = items_dataset.cal_agreement_span(None, agreement_table=prediction[batch_id], min_agree=min_label, max_agree=max_label)
        #decode spans
        for span in spans:
            #print(span)
            if span[0]==0: span[0]+=1
            if span[1]==1: span[1]+=1

            while(True):
              start = test_batch[batch_id].token_to_chars(span[0])
              if start != None or span[0]>=span[1]:
                break
              span[0]+=1
              
            while(True):
              end = test_batch[batch_id].token_to_chars(span[1])
              if end != None or span[0]>=span[1]:
                break
              span[1]-=1

            if span[0]<span[1]:
              de_start = test_batch[batch_id].token_to_chars(span[0])[0]
              de_end = test_batch[batch_id].token_to_chars(span[1]-1)[0]
              #print(de_start, de_end)
              #if(de_start>512): print(de_start, de_end)
              decode_span_table[de_start:de_end]=2 #insite
              decode_span_table[de_start]=1 #begin
        if change_sample:
          sample_result["predict_span_table"] = decode_span_table
          #sample_result["boundary"] = test_batch["boundary"][id]
          result.append(sample_result)
  model.train()
  return result
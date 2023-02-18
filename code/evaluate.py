import torch

from args import args, config
from tqdm import tqdm
from collections import Counter
def sentence_level_predict(batch_id, start, end, prediction, threshold_num, threshold_rate):
  """"""
  if start>510:
    print("to long")
    return -1
  if end>510: end=510
  cnt = Counter()
  check_worthy_num=0
  non_check_worthy_num=0
  for label in prediction[batch_id][start:end]:
     if label==0: non_check_worthy_num+=1
     else: check_worthy_num+=1
  if check_worthy_num >= ((end-start)*threshold_rate):
    return 1
  elif check_worthy_num >= threshold_num:
    return 1
  else:
    return 0

def renew_confution_matrix(ground_truth, predict_label, confution_matrix):
  if ground_truth==0 and predict_label==1:
    confution_matrix["FP"]+=1
  elif ground_truth==0 and predict_label==0:
    confution_matrix["TN"]+=1
  elif ground_truth==1 and predict_label==0:
    confution_matrix["FN"]+=1
  elif ground_truth==1 and predict_label==1:
    confution_matrix["TP"]+=1
  else:
    print("error")

def test_encode_predict(test_loader, device, model):
  model.eval()
  predict_list = []
  ground_truth = []
  padding_mask_list = []
  for i, test_batch in enumerate(tqdm(test_loader)):
    #batch_text = test_batch['batch_text']
    input_ids = test_batch['input_ids'].to(device)
    token_type_ids = test_batch['token_type_ids'].to(device)
    attention_mask = test_batch['attention_mask'].to(device)
    ground_truth.extend(test_batch['labels'].to(device))
    
    crf_mask = test_batch["crf_mask"].to(device)
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=None, crf_mask=crf_mask)
    if args.use_crf:
      prediction = model.crf.decode(output[0], crf_mask)
    else:
      prediction = torch.max(output[0], -1).indices
    predict_list.extend(prediction)
    padding_mask_list.extend(crf_mask)
  model.train()
  return predict_list, ground_truth, padding_mask_list

def cal_span_level_matrix(predict_list, ans_list, padding_mask_list, begin_match=False):
  confution_matrix ={"TP":0, "FP":0, "FN":0, "TN":0}
  for sample_id in range(len(ans_list)):
    for pre_word, ans_word, padding_mask in zip(predict_list[sample_id], ans_list[sample_id], padding_mask_list[sample_id]):
      if not begin_match:
        if pre_word==2: pre_word=1
        if ans_word==2: ans_word=1
      if padding_mask: renew_confution_matrix(ans_word, pre_word, confution_matrix)
  return confution_matrix

def calculate_precision_recall_f1(confution_matrix):
  """
  input: confution_matrix)({"TP":0, "FP":0, "FN":0, "TN":0})
  output: {"precision":(float), "recall":(float), "f1":(float), "accuracy":(float)}
  """
  TP, FP, FN, TN = (confution_matrix["TP"], confution_matrix["FP"], confution_matrix["FN"], confution_matrix["TN"])
  result = dict()
  if (TP + FP)!=0:
    result["precision"] = TP / (TP+FP)
  else:
    result["precision"] = 1.0
  if (TP + FN)!=0:
    result["recall"] = TP / (TP+FN)
  else:
    result["recall"] = 1.0
  if (result["precision"]+result["recall"])!=0:
    result["f1"] = 2*(result["precision"]*result["recall"])/(result["precision"]+result["recall"])
  else:
    result["f1"] =0.0
  if (TP + FN + TN + FP)!=0:
    result["accuracy"] = (TP+TN) / (TP + FN + TN + FP)
  else:
    result["accuracy"] = 1.0
  if ((TP+FP) * (TN+FN) * (TP+FN) * (TN+FP))!=0:
    result["mcc"]=((TP*TN)-(FP*FN))/(((TP+FP) * (TN+FN) * (TP+FN) * (TN+FP))**0.5)
  else:
    result["mcc"]=1.0
  return result

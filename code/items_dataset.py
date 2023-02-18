import torch
from torch.utils.data import Dataset
from args import args
class items_dataset(Dataset):
    def __init__(self, tokenizer, data_set, label_dict, max_length=args.max_length):
        self.data_set = data_set
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.max_length = max_length
        self.encode_max_length = max_length-2 #[CLS] [SEP]
        self.batch_max_lenght = max_length

    def __getitem__(self, index):
        result = self.data_set[index]
        return result
    
    def __len__(self):
        return len(self.data_set)

    def create_label_list(self, span_label, max_len):
      #ans = []
      table = torch.zeros(max_len)
      for start, end in span_label:
        table[start:end] = 2 #"I"
        table[start] = 1 #"B"
      """
      for label in table.tolist():
        if label == 0:
            ans.append("O")
        elif label == 1:
            ans.append("B")
        elif label == 2:
            ans.append("I")
        else:
            print("error")
      """
      return table
    def encode_lable(self, encoded, batch_table):
      batch_encode_seq_lens = []
      sample_mapping = encoded["overflow_to_sample_mapping"]
      offset_mapping = encoded["offset_mapping"]
      encoded_label = torch.zeros(len(sample_mapping) ,self.encode_max_length, dtype=torch.long)
      for id_in_batch in range(len(sample_mapping)):
        encode_len=0
        table = batch_table[sample_mapping[id_in_batch]]
        for i in range(self.max_length):
          char_start, char_end = offset_mapping[id_in_batch][i]
          # ignore [CLS], [SEP] token
          if char_start!=0 or char_end!=0:
              encode_len+=1
              #print(encoded_label.shape, table.shape)
              encoded_label[id_in_batch][i-1] = table[char_start].long()
        batch_encode_seq_lens.append(encode_len)
      return encoded_label, batch_encode_seq_lens

      
    def create_crf_mask(self, batch_encode_seq_lens):
        mask = torch.zeros(len(batch_encode_seq_lens), self.encode_max_length, dtype=torch.bool)
        #print(len(batch_table), len(batch_lens), seq_lens, batch_lens)
        for i, batch_len in enumerate(batch_encode_seq_lens):
            mask[i][:batch_len]=True
        return mask
    
    def boundary_encoded(self, encodings, batch_boundary):
      batch_boundary_encoded = []
      for batch_id, span_labels in enumerate(batch_boundary):
        boundary_encoded = []
        end = 0
        for boundary in span_labels:
          end += boundary

          encoded_end = encodings[batch_id].char_to_token(end-1)
          
          #
          tmp_end = end
          while encoded_end==None and tmp_end>0:
            tmp_end-=1
            encoded_end = encodings[batch_id].char_to_token(tmp_end-1)
          if end!=None: encoded_end+=1

          if encoded_end>self.encode_max_length:
            boundary_encoded.append(self.encode_max_length)
            break
          else:
            boundary_encoded.append(encoded_end)
        for i in range(len(boundary_encoded)-1, 0, -1):
          boundary_encoded[i]=boundary_encoded[i]-boundary_encoded[i-1]
            
        batch_boundary_encoded.append(boundary_encoded)
      return batch_boundary_encoded
    def cal_agreement_span(self, agreement_table, min_agree=2, max_agree=3):
      """
      find the spans from agreement table
      """
      ans_span=[]
      start, end =(0, 0)
      pre_p = agreement_table[0]
      for i, word_agreement in enumerate(agreement_table):
        curr_p = word_agreement
        if curr_p != pre_p:
          if start != end: ans_span.append([start, end])
          start=i
          end=i
          pre_p = curr_p
        if word_agreement<min_agree:
          start+=1
        if word_agreement<=max_agree:
          end+=1
        #print([start, end])
        pre_p = curr_p
      if start != end: ans_span.append([start, end])
      #print(ans_span)
      if len(ans_span)<=1 or min_agree == max_agree:
        return ans_span
      #span 合併
      span_concate = []
      start, end = [ans_span[0][0], ans_span[0][1]]
      for span_id in range(1, len(ans_span)):
        if ans_span[span_id-1][1]==ans_span[span_id][0]:
          ans_span[span_id]=[ans_span[span_id-1][0], ans_span[span_id][1]]
          if span_id==len(ans_span)-1: span_concate.append(ans_span[span_id])
          #span_concate.append()
        elif span_id==len(ans_span)-1:
          span_concate.extend([ans_span[span_id-1], ans_span[span_id]])
        else:
          span_concate.append(ans_span[span_id-1])
      return span_concate

    def collate_fn(self, batch_sample):
        batch_text = []
        batch_table = []
        batch_span_label= []
        seq_lens = []
        for sample in batch_sample:
          batch_text.append(sample['original_text'])
          batch_table.append(self.create_label_list(sample["span_labels"], len(sample['original_text'])))
          #batch_boundary = [sample['data_len_c'] for sample in batch_sample]
          batch_span_label.append(sample["span_labels"])
          seq_lens.append(len(sample['original_text']))
        self.batch_max_lenght = max(seq_lens)
        if self.batch_max_lenght > self.encode_max_length : self.batch_max_lenght = self.encode_max_length 

        encoded = self.tokenizer(batch_text, truncation=True, max_length=512, padding='max_length', stride=25, return_overflowing_tokens=True, return_tensors="pt", return_offsets_mapping=True)
        #encoded = self.tokenizer(batch_text, truncation=True, padding=True, return_tensors="pt", max_length=self.max_length)
        
        encoded['labels'], batch_encode_seq_lens = self.encode_lable(encoded, batch_table)
        encoded["crf_mask"] = self.create_crf_mask(batch_encode_seq_lens)
        #encoded["boundary"] = batch_boundary
        #encoded["boundary_encode"] = self.boundary_encoded(encoded, batch_boundary)
        encoded["span_labels"] = batch_span_label
        encoded["batch_text"] = batch_text
        return encoded
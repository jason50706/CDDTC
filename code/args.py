class args():
    DATA_PATH = "../Dataset/"
    SAVE_MODEL_PATH = "../model/"
    
    #pre_model_name = "bert-base-chinese"
    #pre_model_name = "hfl/chinese-macbert-base"
    pre_model_name = "hfl/chinese-roberta-wwm-ext"
    save_model_name = "roberta_crf"
    
    LOG_DIR = "../log/long_term/"+save_model_name+"/"
    
    use_crf = True
    label_dict = {"O":0, "B":1, "I":2}
    epoch_num = 10
    batch_size = 2
    label_size = 3
    max_length = 512

class config():
    hidden_dropout_prob = 0.1
    hidden_size = 768
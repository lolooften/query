from train import Data, SeqLabel, batchify_sequence_labeling_with_label, recover_label
import torch
import os
from contextlib import redirect_stdout
import io
TMP_FILE = 'input.tmp'

def form_pairs(sentences, file_path=TMP_FILE):
    '''
    将读入的自然语言句子加入空标签, 并按每个字符单独占一行、句与句之间用空行分隔的方式格式化.
    '''
    fout = open(file_path, 'w', encoding='utf-8')
    for sentence in sentences:
        tokens = list(sentence)
        for token in tokens:
            print(token + ' O', file=fout)
        print('\r\n', file=fout)
    fout.close()
    return file_path

def predict_ne(dset_dir, model_dir, input_text):
# unsilence command-line output

    # 虽然 raw 中有 label, 但是做预测的时候没有用到, 只有 evaluate 时用到了, 可以全部指定为 O
    data = Data()
    data.HP_gpu = torch.cuda.is_available()

    data.load(dset_dir)

    input_file = form_pairs(input_text)
    
    data.raw_dir = input_file

    data.generate_instance('raw')

    # 暂时停止在 SeqLabel 读入时的输出
    with redirect_stdout(io.StringIO()) as f:
        model = SeqLabel(data)

    model.load_state_dict(torch.load(model_dir))

    instances = data.raw_Ids
    
    pred_results = []
    model.eval()
    batch_size = data.HP_batch_size
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_sequence_labeling_with_label(instance, data.HP_gpu, False)
        tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        pred_label, _ = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label

    os.remove(input_file)
    
    return [raw_text[0] for raw_text in data.raw_texts], pred_results

def recover_entities(raw_texts, pred_results):
    entities_list = []
    for (raw_text, pred_result) in zip(raw_texts, pred_results):
        entities = []
        entity = ''
        entity_type = ''
        for (text, label) in zip(raw_text, pred_result):
            if label.startswith('B'):
                if len(entity) != 0:
                    entities.append((entity_type, entity))
                entity_type = label.split('-')[1]
                entity = text
            elif label.startswith('I'):
                if len(entity) != 0:
                    entity += text
            elif label.startswith('O'):
                if len(entity) != 0:
                    entities.append((entity_type, entity))
                    entity = ''
                    entity_type = ''
        if len(entity) != 0:
            entities.append((entity_type, entity))
        entities_list.append(entities)
    return entities_list


if __name__ == '__main__':

    recover_entities([['a',  'b',  'c',  'd','e',  'f',  'g',  'h','i','j',  'k','l',  'm',  'n']],
                     [['B-a','B-a','I-a','O','B-b','I-b','I-b','O','O','B-c','O','B-d','I-d','B-e']])
    print(0)
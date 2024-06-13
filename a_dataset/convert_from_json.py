'''
convert_from_json.py

读入结构化的 .json 数据集, 生成用于后续处理的结构化实例 .txt 文本文件、用于 NeuralNLP 分类的 .json 文件、用于 NER 的 .txt 文件.
生成 sql 在数据库中的执行结果文件, 用于 result_match.

在 NeuralNLP 中的分类任务有 3 个:
    --- 识别 4 个数据表中用到了哪些表.
    --- 识别出现了哪些与时间相关的函数 (例如 addtime, adddate, curdate, now, date, timediff).
    --- 识别位号、装置、单位.
需要的文件:
    --- dataset.json (没有用到 dataset_gbk.json)
生成的文件:
    --- data_train.txt & data_test.txt (普遍用途, seq2seq 模型, 访问 sql 数据库)
    --- data_all.txt (合并了 data_train.txt 和 data_test.txt 的数据, 用于检验)
    --- data_canonical.txt (每一个问题原型只产生一条语句)
    --- data_table_train.json & data_table_test.json (识别用到哪些表)
    --- data_time_train.json & data_time_test.json (识别用到哪些时间相关函数)
    --- data_ner_train.txt.ann & data_ner_test.txt.ann (识别位号、装置、单位)
'''

from __future__ import print_function
import json
import codecs
import config
import transformers
import re
import sys
sys.path.append('YEDDA')
from YEDDA import getWordTagPairs


def get_used_keywords(sql, keyword_list):
    used_keywords = []
    for keyword in keyword_list:
        if bool(re.search(keyword, sql)):
            used_keywords.append(keyword)
    if len(used_keywords) == 0:
        used_keywords.append('False')
    else:
        used_keywords.append('True')
    return used_keywords


def tokenize(tokenizer, text, wordwise=False):
    if wordwise == True:
        pass
    else:
        return tokenizer.tokenize(text)


def write_pair(sentence_list, file_name):
    # 由 YEDDA.py 的 generateSequenceFile(self) 函数中修改而来
    seq_file = open(file_name, 'w', encoding='utf-8')
    for sentence in sentence_list:
        if len(sentence) <= 2:
            seq_file.write('\n')
            continue
        else:
            word_tag_pairs = getWordTagPairs(tagedSentence=sentence, segmented=False, tagScheme='BIO', onlyNP=False)
            for wordTag in word_tag_pairs:
                seq_file.write(wordTag)
            # use null line to separate sentences
            seq_file.write('\n')
    seq_file.close()


if __name__ == '__main__':
    query_split = True
    tokenizer = transformers.BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    output_prefix = 'dataset_query_based_split/data' if query_split else 'dataset/data'
    keyword_list = ['addtime', 'adddate', 'curdate', 'date', 'now', 'timediff']
    
    data = json.loads(open(config.path_dataset).read())
    out_flat_train = open(output_prefix + '_train.txt', 'w', encoding='utf-8')
    out_flat_test = open(output_prefix + '_test.txt', 'w', encoding='utf-8')
    out_flat_all = open(output_prefix + '_all.txt', 'w', encoding='utf-8') # 所有数据, 用来校验数据集有没有问题
    out_flat_canonical = open(output_prefix + '_canonical.txt', 'w', encoding='utf-8') # 所有问题原型的单个数据, 用来校验数据集有没有问题
    
    # 生成的 .ann 文件通过 YEDDA 转化为 BIO 标注的 .anns 文件
    out_ner_train = []
    out_ner_test = []
    
    out_table_train = codecs.open(output_prefix + '_table_train.json', 'w', encoding='utf-8')
    out_table_test = codecs.open(output_prefix + '_table_test.json', 'w', encoding='utf-8')
    out_time_train = codecs.open(output_prefix + '_time_train.json', 'w', encoding='utf-8')
    out_time_test = codecs.open(output_prefix + '_time_test.json', 'w', encoding='utf-8')

    for entry in data:
        data_table_dict = {}
        data_time_dict = {}
        var_sql = entry['sql'][0]
        for i, sentence in enumerate(entry['sentences']):
            text = sentence['text']
            sql = var_sql
            text_anno = sentence['text']
            for name in sentence['variables']:
                value = sentence['variables'][name]
                if type(value) == list: # 替换的是 x, y, u, 这三个变量在 nl 和 sql 中有不同的表达, 因此是 list
                    text = value[0].join(text.split(name))
                    sql = value[1].join(sql.split(name))
                    tmp = '[@' + value[0] + '#' + name.rstrip('0123456789') + '*]' # 即生成像 `[@T202塔底温度#variable*]' 这样的字符串
                    text_anno = tmp.join(text_anno.split(name))
                else: # 其余数值变量在 nl 和 sql 中的表达相同
                    text = value.join(text.split(name))
                    sql = value.join(sql.split(name))
                    text_anno = value.join(text_anno.split(name))

            data_table_dict['doc_label'] = entry['used-tables']
            data_table_dict['doc_token'] = tokenize(tokenizer, text)
            data_table_dict['doc_keyword'] = []
            data_table_dict['doc_topic'] = []

            data_time_dict['doc_label'] = get_used_keywords(sql, keyword_list=keyword_list)
            data_time_dict['doc_token'] = tokenize(tokenizer, text)
            data_time_dict['doc_keyword'] = []
            data_time_dict['doc_topic'] = []

            if query_split:
                output_flat_file = out_flat_train if entry['query-split'] == 'train' else out_flat_test
                output_ner_list = out_ner_train if entry['query-split'] == 'train' else out_ner_test
                output_table_file = out_table_train if entry['query-split'] == 'train' else out_table_test
                output_time_file = out_time_train if entry['query-split'] == 'train' else out_time_test
            else:
                output_flat_file = out_flat_train if sentence['question-split'] == 'train' else out_flat_test
                output_ner_list = out_ner_train if sentence['question-split'] == 'train' else out_ner_test
                output_table_file = out_table_train if sentence['question-split'] == 'train' else out_table_test
                output_time_file = out_time_train if sentence['question-split'] == 'train' else out_time_test
            print(text + '\t' + sql, file=output_flat_file)
            print(text + '\t' + sql, file=out_flat_all)
            if i == 0:
                print(text + '\t' + sql, file=out_flat_canonical)
            output_ner_list.append(text_anno)
            output_table_file.write(json.dumps(data_table_dict, ensure_ascii=False))
            output_table_file.write('\n')
            output_time_file.write(json.dumps(data_time_dict, ensure_ascii=False))
            output_time_file.write('\n')

    out_flat_train.close()
    out_flat_test.close()
    out_flat_all.close()
    out_flat_canonical.close()
    write_pair(out_ner_train, output_prefix + '_ner_train.txt')
    write_pair(out_ner_test, output_prefix + '_ner_test.txt')
    out_table_train.close()
    out_table_test.close()
    out_time_train.close()
    out_time_test.close()

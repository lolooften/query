'''
generate_output_vocab.py

根据输入的 nl 句子判断在生成 sql 语句时的可用词表.

根据输入的句子判断:
    1. 用到哪些表 (如 t_rt_data 等 table), 从而将表中的列名加入可用词表
    2. 用到哪些关键词 (adddate 等)
    3. 将数字、字符等添加至词表中 (PointerNetwork 的 copy 机制)
    4. NER 将自然语言文本, 特别是关于 id 这样的出现次数很少的信息添加至词表 (思想是 ValueNet 的思想)

请注意, 在可拓展性方面, 只有 4 是需要人工给定列表的, 但此问题具有一定普遍意义, 即关于 id-描述 的数量很大, 难以通过一般方法小规模训练.
3 和 4 本质上的作用是一样的, 都是从大范围的候选词中选择符合的加入词表.

1 和 2 是文本分类问题, 按特定函数关键词是否出现分类.
3 和 4 是命名实体识别, 因为有一一对应关系. 其中 3 在 nl 中与 sql 中形式相同, 而 4 在两种语言中形式不同.
'''

from pyparsing import *
import re
import os
import json
import sys
import transformers
import jellyfish
import numpy as np
import hashlib
from manipulate_cache import load_cache_from_file, save_cache_to_file

tokenizer = transformers.BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')

vocab_cache = load_cache_from_file('global_vocab_cache.pkl')

def vocab_search_or_predict(texts):
    # texts 需要为列表
    if type(texts) is str:
        texts = [texts]
    # 对于句子列表, 检查其中的每个句子是否都有相应的预测词汇结果
    # 如果有, 记录它的编号, 并不再重复预测
    # 如果没有, 合并这些没有的句子成一个新的列表, 并统一进行预测并保存结果
    n = len(texts)
    predict_label_list = [0 for _ in range(n)]
    vocab_list = [[] for _ in range(n)]
    text_list_new = []
    for i, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in vocab_cache:
            vocab_list[i] = vocab_cache[text_hash]
        else:
            predict_label_list[i] = 1
            text_list_new.append(text)
    if len(text_list_new) > 0:
        vocab_list_new = predict_vocab(text_list_new)
        for i in range(n):
            if predict_label_list[i] == 1:
                text = texts[i]
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                vocab_list[i] = vocab_list_new.pop(0)
                vocab_cache[text_hash] = vocab_list[i]
        save_cache_to_file(vocab_cache, 'global_vocab_cache.pkl')
    return vocab_list


def predict_vocab(texts):
    
    # 后续处理需要 texts 是句子列表
    if isinstance(texts, str):
        texts = [texts]

    # 保存当前工作路径
    wd = os.getcwd()

    # 本文件工作路径
    this_wd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(this_wd)

    ## 1. tables & 2. keywords
    # 导入 predict 函数, 由于函数初始化过程用到了同一文件夹的文件, 需要将 predict 所在文件夹加入路径
    sys.path.append('b_table_classification/NeuralNLP')
    from b_table_classification.NeuralNLP.predict import predict

    # 需要将工作路径改为 NeuralNLP 所在路径, 因为 NeuralNLP 的预测用到了 globals()
    os.chdir('b_table_classification/NeuralNLP')
    config_dir = 'conf/'
    config_files = [config_dir + f for f in os.listdir(config_dir)]

    # 表结构
    table_structure = {
        't_rt_info': ['Tag_code', 'Tag_desc', 'Tag_unit', 'Key_tag', 'Unit_code',
                      'LLower_limit', 'Lower_limit', 'Upper_limit', 'UUpper_limit'],
        't_rt_data': ['Id', 'Tag_code', 'Result_value', 'Result_time', 'Alarm_status'],
        't_model_data': ['Tag_code', 'Inference_path', 'Inference_root_cause_id'],
        't_inference_data': ['Id', 'Rt_code', 'Inference_time', 'Inference_path',
                             'Inference_score', 'Inference_root_cause_id', 'Inference_root_cause_status'],
    }

    # texts 长度至少为 2, 否则会在 squeeze 的时候被压缩维度报错
    texts_pad_flag = 0
    if len(texts) < 2:
        texts.append(texts[0]) # 复制一份
        texts_pad_flag = 1
    
    tokens_list = [tokenizer.tokenize(text) for text in texts]

    # 使用 labels 字典存储文本的关键词
    labels = {}

    for config_file in config_files:
        # 根据文件名读取配置文件
        config = json.load(open(config_file, 'r'))

        # 对每个配置文件分别预测
        labels[config_file] = predict(config_file=config_file, tokens_list=tokens_list)

        # 处理关键词
        # hierarchical 在这里被用来处理 null label 的情况, 因为多分类不支持 null label
        # 第一级标签是 True 和 False, True 代表非 null label, False 代表 null label
        # True 下有第二级标签, 是真正的 label, False 没有二级标签
        # 处理时删去 True 和 False 即可
        if config['task_info']['hierarchical'] is True:
            for i, _ in enumerate(labels[config_file]):
                if 'True' in labels[config_file][i]:
                    labels[config_file][i].remove('True')
                if 'False' in labels[config_file][i]:
                    labels[config_file][i].remove('False')

        # 处理表结构, 配置文件中带有 `table', 并且应该是唯一的配置文件
        # 将匹配到的表中的列名加入
        if bool(re.search(r'table', config_file)):
            for i, tables in enumerate(labels[config_file]):
                tmp = []
                for table in tables:
                    tmp.extend([table + '.' + column for column in table_structure[table]])
                labels[config_file][i].extend(tmp)
    # 改回本文件工作路径
    os.chdir(this_wd)
    sys.path.remove('b_table_classification/NeuralNLP')


    ## 3. NER - heuristics
    # 这部分 NER 做的是在 sql 与 nl 中形式相同的, 暂时只有数字 (分钟、小时、天数、数量等)
    # 位号名中，有些是字母开头的，这些需要舍去, 因此只匹配两端都是中文字符 [\u4e00-u9fff] 的数字
    # 为了匹配符号, 放开到 \u2e80
    # 中文标点如下 (对应 \uff01 开始的全宽字符)
    # ff01 -> ！
    # ff08 -> （
    # ff09 -> ）
    # ff0c -> ，
    # ff1a -> ：
    # ff1b -> ；
    # ff1f -> ？
    rule_numeral = r'(?<![A-Za-z])[+-]?\d*[.]?\d+(?![A-Za-z])'
    result_numeral = [re.findall(rule_numeral, text) for text in texts]
    labels['NER_heuristic'] = result_numeral


    ## 4. NER - bert-bilstm-crf
    # 这部分 NER 做的是在 sql 与 nl 中形式不同的, 命名实体对应着生成数据集时所用的 x (位点), y (装置), u (单位)
    # 删除加入的路径 (b_table_classification/NeuralNLP), 并加入新路径
    sys.path.append('c_ner/NCRFpp')
    if 'model' in sys.modules.keys():
        del sys.modules['model']
    from c_ner.NCRFpp.predict_ne import predict_ne, recover_entities

    raw_texts, pred_results = predict_ne(dset_dir='c_ner/NCRFpp/data/saved_model.lstmcrf.dset',
                                         model_dir='c_ner/NCRFpp/data/saved_model.lstmcrf',
                                         input_text=texts)
    # 以 (类别, 命名实体) 的形式存储识别出的结果
    entities_list = recover_entities(raw_texts, pred_results)
    # 将命名实体转化为 sql 语句中的形式, 例如 `一催' -> `3B1CHLH'
    # 首先读入预置的 nl -> sql 转化的列表
    def get_pair_from_file(file_path, sep=','):
        pair_list = []
        f = open(file_path, 'r', encoding='utf-8')
        for line in f:
            text, sql = line.rstrip().split(sep)[:2] # x.txt 中包含 3 列, 最后一列是 dcs_code
            pair_list.append((text, sql))
        return pair_list

    replace_file_dict = {
        'variable': get_pair_from_file('a_dataset/dataset_raw/x.txt'),
        'device': get_pair_from_file('a_dataset/dataset_raw/y.txt'),
        'unit': get_pair_from_file('a_dataset/dataset_raw/u.txt'),
        }
    # 使用 jellyfish 比较相似性
    entities_list_sql = []
    for entities in entities_list:
        entities_sql = []
        for entity in entities:
            label, text = entity
            # jellyfish 的 levenshtein 距离比较的是字符串间的距离, 而不是列表
            # 如果 entity label 没有被 NER 识别出来, 而是空 (会报错), 那就跳过
            try:
                sim_scores = [jellyfish.levenshtein_distance(text, target[0]) for target in replace_file_dict[label]]
            except:
                continue
            min_index = np.argmin(sim_scores)
            # 因为在 sql 语句中, 这些词需要加引号
            entities_sql.append("'" + replace_file_dict[label][min_index][1] + "'")
        entities_list_sql.append(entities_sql)
    sys.path.remove('c_ner/NCRFpp')
    labels['NER_ml'] = entities_list_sql

    ## 总结所有关键词

    punctuations = [',', '*', ';', '(', ')', ':', "'",
                    '-',
                    '=', '>', '<', '>=', '<=', '!=',
                ]
    
    keyword_others = ['0', '1', '2'] # 例如用来表示对错、报警分类状态, 0 还经常用在时间表示中占位

    keyword_nonfunctions = [
        'SELECT', 'FROM', 'WHERE', 'DISTINCT',
        'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'IS', 'NULL',
        'INNER', 'LEFT', 'JOIN', 'ON', 'GROUP', 'AS', 'ORDER', 'BY', 'LIMIT', 'HAVING', 'ASC', 'DESC',
    ]

    keyword_functions = [
        'AVG', 'MAX', 'MIN', 'SUM', 'COUNT',
        'LENGTH', 'ISNULL',
    ]

    vocab_list = []
    for i in range(len(labels[list(labels)[0]])):
        tokens = []
        for key in list(labels):
            tokens += labels[key][i]
        tokens += keyword_nonfunctions
        tokens += keyword_functions
        tokens += keyword_others
        tokens += punctuations
        # 将所有都改成小写
        tokens = [token.lower() for token in tokens]
        vocab_list.append(tokens)

    # 改回原来的工作路径
    os.chdir(wd)
    
    if texts_pad_flag:
        # 说明原来只有 1 句句子, 经过了 padding, 因此输出时需要去掉
        return vocab_list[:-1]
    else:
        return vocab_list


if __name__ == '__main__':
    # 待生成候选词的句子
    # texts = []
    # with open('a_dataset/dataset/data_test.txt', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         texts.append(line.rstrip())
    texts = ['告诉我根原因是装置连续重整装置1当中的变量的根原因诊断结果，按它们的单位分类输出。']
    v = predict_vocab(texts)
    print(0)
    
'''
structuralize.py

读入半结构化的 .txt.ann 数据集, 生成如 `geography.json' 式的结构化的 .json 数据集
需要的文件:
    --- dataset.txt.ann (使用 Yedda 标注了模板问题中可替换部分位置的 dataset.txt 文件)
    --- x.txt, y.txt, u.txt (分别是可替换的位号、装置、单位, 其中 x.txt 来自于 t_rt_info.xlsx)
    --- aux_vars.json (里面包含了数据库的信息, 在实例化时需要考虑数据库, 生成贴合数据库实际的问题)
生成的文件:
    --- dataset.json (utf-8 编码)
    --- dataset_gbk.json (gbk 编码，仅为了方便查看, 在后续程序中没有用到)
'''

import re
import config
import numpy as np
import random
import json
import codecs

dataset = [] # 所有问题列表
train_split = .8 # 在 query-based split 中的训练集占比
used_tables = [] # 问题使用的表
n_mapping = lambda x: 1 if x==0 else 2*x # 决定每个原型问题按照出现的参数个数 (x) 进行多少次实例化
nl_lines = [] # 当前 nl 问题 (可能有多个)
data_id = 0 # 问题编号
random.seed(0)

# 将 sql 中的括号、逗号等加空格,
# 可以将规则最后 5 行取消注释
def process_sql(line):
    replacements = [(r'\s*\(\s*', ' ( '),
                    (r'\s*\)\s*', ' ) '),
                    (r'\s*,\s*', ' , '),
                    (r'\s*;\s*', ' ;'),
                    (r'\s*-\s*', ' - '),
                    (r"\s*'\s*", " ' "),
                    (r'\s*:\s*', ' : '),
                    # (r'(?<=\w)\s+\(', '('),
                    # (r'\(\s+\)', '()'),
                    # (r'select\(', 'select ('),
                    # (r'^\s*\(', '('),
                    # (r'\sin\(', ' in ('),
                    ]
    for old, new in replacements:
        line = re.sub(old, new, line)
    return line


## 每个领域的实例化列表
# 读取存储 (variable, device, unit) 的文本文件
def get_instantiation_pair(file):
    pair_list = []
    for line in open(file, 'r', encoding='utf-8'):
        line = line[:-1] # 去掉每行最后的 `\n'
        nl, sql = line.split(',')[:2]
        sql = '\'' + sql + '\''
        pair_list.append((nl, sql))
    return pair_list
variable_list = get_instantiation_pair(config.path_variable)
device_list = get_instantiation_pair(config.path_device)
unit_list = get_instantiation_pair(config.path_unit)

# 在 t_inference_data 和 t_model_data 中出现过的相关变量
aux_vars_dict = json.loads(open(config.path_aux_vars).read())
variable_list_inference = list(filter(lambda x: x[1].split("'")[1] in aux_vars_dict['t_inference_data_vars'], variable_list))
variable_list_model = list(filter(lambda x: x[1].split("'")[1] in aux_vars_dict['t_model_data_vars'], variable_list))

field_replace_dict = {
    'time_hour': np.r_[1:13].tolist(),
    'time_minute': np.r_[0:60].tolist(),
    'minutes': np.r_[1:60].tolist(),
    'hours': np.r_[1:12, 12:36:6, 36:132:12].tolist(),
    'days': np.r_[1:31].tolist(),
    'count': np.r_[1:10, 10:10:60, 100, 200].tolist(), # 和数据库的条目数相关的用 count (例如和 limit 关键字联用), 否则用 value
    'variable': variable_list,
    'variable2': variable_list,
    'device': device_list,
    'device2': device_list,
    'unit': unit_list,
    'value': np.r_[1:21].tolist(),
}

## 读取存储半结构化数据集的文本文件
for line in open(config.path_text, 'r', encoding='utf-8'):
    line = line[:-1] # 去掉每行最后的 `\n'
    # 文本中以 `#' 开头的行被忽略
    if line.startswith('#'):
        continue
    elif line.startswith('*******'):
        # 数据集中以 `*******' 开头的行代表下面的数据用到哪些表
        used_tables = re.split(r'\s*[*]+\s*', line)[1].split(' + ')
    elif bool(re.search(r'[\u4e00-\u9fa5]', line)):
        # 包含中文字符，说明仍然是自然语言问题，需要继续读行
        nl_lines.append(line)
        continue
    else:
        # 此分枝是 sql 
        # 将 sql 语句中的 `Dcs_code' 全部替换为 `Tag_code'
        line = re.sub('Dcs_code', 'Tag_code', line)
        line = process_sql(line) # 调整特殊字符前后的空格
        query_split = 'train' if random.random() < train_split else 'test'
        sentences = []
        variables = []

        for field in list(field_replace_dict):
            # 对每个领域的标记进行匹配
            pattern = r'\[@[^#]*?#{field}\*\]'.format(field=field)
            matched = re.findall(pattern, line)
            matched_unique = list(dict.fromkeys(matched)) # 去除 matched 的重复值, 同时保持顺序
            # 对每个标记进行抽象化
            for i, v in enumerate(matched_unique):
                line = line.replace(v, field + str(i))
                for j, nl_line in enumerate(nl_lines):
                    nl_lines[j] = nl_lines[j].replace(v, field + str(i))
                variables.append({
                    'example': str(re.findall(r'(?<=@).*(?=#)', v)[0]),
                    'location': 'both',
                    'name': field + str(i),
                    'type': field,
                })
        # 对句子进行实例化
        for j, nl_line in enumerate(nl_lines):
            question_split = 'train' if j % 2 == 0 else 'test'
            for _ in range(n_mapping(len(variables))):
                variables_nl = {}
                for v in variables:
                    tmp = random.choice(field_replace_dict[v['type']])
                    # 对于 (nl-form, sql-form) 即在 nl 和 sql 中表述不同的二元组的变量, 不用转化为字符串
                    variables_nl[v['name']] = tmp if type(tmp) == tuple else str(tmp)
                ## 处理特殊情况
                # 目前数据集中, 只有 time_minute, time_hour, value, count 会有在同一个句子中多于 1 个并且需要遵循大小关系的情形,
                # 例如:
                    # `8 点 10 分到 9 点 20 分' (time_minue, time_hour)
                    # `8 点到 9 点和 9 点到 10 点' (time_hour)
                    # `低报阈值小于 5 并且高报阈值大于 20' (value)
                    # `次数位于 0 到 10 之间' (count)
                # 需要保证出现的顺序和数字大小顺序一致 (目前 time_minute 不考虑大小顺序)
                if 'time_hour1' in variables_nl:
                    if 'time_hour2' in variables_nl:
                        tmp = random.sample(field_replace_dict['time_hour'], 3)
                        tmp.sort()
                        variables_nl['time_hour0'] = str(tmp[0])
                        variables_nl['time_hour1'] = str(tmp[1])
                        variables_nl['time_hour2'] = str(tmp[2])
                    else:
                        tmp = random.sample(field_replace_dict['time_hour'], 2)
                        tmp.sort()
                        variables_nl['time_hour0'] = str(tmp[0])
                        variables_nl['time_hour1'] = str(tmp[1])
                if 'value1' in variables_nl:
                    tmp = random.sample(field_replace_dict['value'], 2)
                    tmp.sort()
                    variables_nl['value0'] = str(tmp[0])
                    variables_nl['value1'] = str(tmp[1])
                if 'count1' in variables_nl:
                    tmp = random.sample(field_replace_dict['count'], 2)
                    tmp.sort()
                    variables_nl['count0'] = str(tmp[0])
                    variables_nl['count1'] = str(tmp[1])
                
                # value 和 variable 同时出现, 说明两者有关
                # 实例不多, 暂不考虑实现
                
                # 由于 t_model_data 中的数据量多于 t_inference_data, 因此先判断 t_inference_data
                # 如果 used_tables 中包含 t_inference_data, 则 variable 应该是表中出现过的变量
                if 'variable0' in variables_nl:
                    if 't_inference_data' in used_tables:
                        variables_nl['variable0'] = random.choice(variable_list_inference)
                    # 如果 used_tables 中包含 t_model_data, 则 variable 应该是表中出现过的变量
                    elif 't_model_data' in used_tables:
                        variables_nl['variable0'] = random.choice(variable_list_model)
                sentences.append({
                    'question-split': question_split,
                    'text': nl_line,
                    'variables': variables_nl,
                })
        entry = {'id': data_id,
                 'query-split': query_split,
                 'sentences': sentences,
                 'sql': [line],
                 'variables': variables,
                 'used-tables': used_tables}
        # 检查是否划定的表都在 sql 语句中出现过
        # for used_table in entry['used-tables']:
        #     if not bool(re.search(used_table, line)):
        #         print('*********', data_id, used_table, sentences)
        dataset.append(entry)
        nl_lines = []
        data_id += 1
json.dump(dataset, codecs.open(config.path_dataset_gbk, 'w', encoding='utf-8'), indent=4, ensure_ascii=False) # 中文字符格式
json.dump(dataset, open(config.path_dataset, 'w', encoding='utf-8'), indent=4) # utf-8 编码格式

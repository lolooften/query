'''
corpus_stats_new.py

仿照 text2sql-data-master\\tools\\corpus_stats.py 生成对于数据集的统计

需要的文件:
    --- dataset.json

统计的信息:
    --- NL 数量、SQL 数量、平均 NL 长度、平均 SQL 长度
    --- NL 长度与 SQL 长度的相关系数
    --- SQL 中平均函数、变量、关键词、不同关键词的数量
'''

import numpy as np
import matplotlib.pyplot as plt
import json
import re

def update_in_quote(in_quote, token):
    if '"' in token and len(token.split('"')) % 2 == 0:
        in_quote[0] = not in_quote[0]
    if "'" in token and len(token.split("'")) % 2 == 0:
        in_quote[1] = not in_quote[1]

keyword_nonfunctions = [
    'SELECT', 'FROM', 'WHERE', 'DISTINCT',
    'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'IS', 'NULL',
    'INNER', 'LEFT', 'JOIN', 'ON', 'GROUP', 'ORDER', 'BY', 'LIMIT', 'HAVING', 'ASC', 'DESC',
]

keyword_functions = [
    'AVG', 'MAX', 'MIN', 'SUM', 'COUNT',
    'LENGTH', 'ISNULL',
]

keyword_mysql_functions = [
    'ADDTIME', 'ADDDATE', 'CURDATE', 'DATE', 'NOW', 'TIMEDIFF',
]

file_name = 'dataset\\dataset.json'
question_space_split = False

# file_name = 'D:\\tmp\\query\\text2sql-data-master\\data\\wikisql.json'
# question_space_split = True

dataset = json.load(open(file_name))
query_count = len(dataset)
stat_dict = {
    'query_count': query_count,
    'correlation': 0,
    'question_count': np.zeros(query_count, dtype=int),
    'query_len': np.zeros(query_count, dtype=int),
    'question_len': [[] for _ in range(query_count)],
    'table_count': np.zeros(query_count, dtype=int),
    'variable_count': np.zeros(query_count, dtype=int),
    'keyword_count': np.zeros(query_count, dtype=int),
    'function_count': np.zeros(query_count, dtype=int),
    'distinct_keyword_count': np.zeros(query_count, dtype=int),
}
for i, data in enumerate(dataset):
    stat_dict['question_count'][i] = len(data['sentences'])
    tmp = data['sql'][0]
    tmp = 'COUNT ('.join(tmp.split('COUNT('))
    tmp = 'LOWER ('.join(tmp.split('LOWER('))
    tmp = 'MAX ('.join(tmp.split('MAX('))
    tmp = 'MIN ('.join(tmp.split('MIN('))
    tmp = 'SUM ('.join(tmp.split('SUM('))
    tmp = 'YEAR ( CURDATE ( ) )'.join(tmp.split('YEAR(CURDATE())'))
    sql_tokens = tmp.split()
    stat_dict['query_len'][i] = len(sql_tokens)
    for sentence in data['sentences']:
        if question_space_split:
            nl_tokens = sentence['text'].split()
            stat_dict['question_len'][i].append(len(nl_tokens))
        else:
            stat_dict['question_len'][i].append(len(re.sub(r'[a-zA-Z]', '', sentence['text'])))
    if 'used-tables' in data.keys():
        stat_dict['table_count'][i] = len(data['used-tables'])
    stat_dict['variable_count'][i] = len(data['variables'])
    
    in_quote = [False, False]
    keyword_list = []
    for token in sql_tokens:
        if token.upper() in keyword_nonfunctions and (not (in_quote[0] or in_quote[1])):
            stat_dict['keyword_count'][i] += 1
            keyword_list.append(token.upper())
        if token.upper() in keyword_functions+keyword_mysql_functions and (not (in_quote[0] or in_quote[1])):
            stat_dict['function_count'][i] += 1
        update_in_quote(in_quote, token)
    stat_dict['distinct_keyword_count'][i] = len(list(set(keyword_list)))
question_len_expanded = [item for sublist in stat_dict['question_len'] for item in sublist]
tmp = [list(np.repeat(stat_dict['query_len'][i], len(stat_dict['question_len'][i]))) for i in range(len(stat_dict['query_len']))]
query_len_expanded = [item for sublist in tmp for item in sublist]
stat_dict['correlation'] = np.corrcoef(question_len_expanded, query_len_expanded)[0][1]

# plotting
binwidth = 5
fig, axs = plt.subplots(2, 3, tight_layout=True)
axs[0, 0].grid(linestyle=':', color='.75')
axs[0, 0].hist(stat_dict['query_len'], bins=range(0, max(stat_dict['query_len']) + binwidth, binwidth), edgecolor='w')
axs[0, 0].set_xlabel('SQL 语句长度', fontproperties='SimSun', fontsize=14)
axs[0, 0].set_ylabel('计数', fontproperties='SimSun', fontsize=14)
axs[0, 1].grid(linestyle=':', color='.75')
axs[0, 1].hist(question_len_expanded, bins=range(0, max(question_len_expanded) + binwidth, binwidth), edgecolor='w')
axs[0, 1].set_xlabel('NL 问题长度', fontproperties='SimSun', fontsize=14)
axs[0, 1].set_ylabel('计数', fontproperties='SimSun', fontsize=14)
axs[0, 2].grid(linestyle=':', color='.75')
axs[0, 2].scatter(question_len_expanded, query_len_expanded, s=10, alpha=.2, edgecolors='none')
axs[0, 2].set_xlabel('NL 问题长度', fontproperties='SimSun', fontsize=14)
axs[0, 2].set_ylabel('SQL 语句长度', fontproperties='SimSun', fontsize=14)
axs[1, 0].grid(linestyle=':', color='.75')
axs[1, 0].hist(stat_dict['function_count'], bins=range(min(stat_dict['function_count']), max(stat_dict['function_count']) + 1, 1), edgecolor='w')
axs[1, 0].set_xlabel('SQL 语句函数数', fontproperties='SimSun', fontsize=14)
axs[1, 0].set_ylabel('计数', fontproperties='SimSun', fontsize=14)
axs[1, 1].grid(linestyle=':', color='.75')
axs[1, 1].hist(stat_dict['variable_count'], bins=range(min(stat_dict['variable_count']), max(stat_dict['variable_count']) + 1, 1), edgecolor='w')
axs[1, 1].set_xlabel('SQL 语句变量数', fontproperties='SimSun', fontsize=14)
axs[1, 1].set_ylabel('计数', fontproperties='SimSun', fontsize=14)
axs[1, 1].set_xticks([0, 1, 2])
axs[1, 2].grid(linestyle=':', color='.75')
axs[1, 2].hist(stat_dict['keyword_count'], bins=range(min(stat_dict['keyword_count']), max(stat_dict['keyword_count']) + 1, 1), edgecolor='w', label='总数', alpha=1)
axs[1, 2].hist(stat_dict['distinct_keyword_count'], bins=range(min(stat_dict['keyword_count']), max(stat_dict['keyword_count']) + 1, 1), edgecolor='w', label='不同数', alpha=.5)
axs[1, 2].legend(prop={"family":"SimSun", 'size':14})
axs[1, 2].set_xlabel('SQL 语句关键词数', fontproperties='SimSun', fontsize=14)
axs[1, 2].set_ylabel('计数', fontproperties='SimSun', fontsize=14)

print('Dataset: {}'.format(file_name.split('\\')[-1].split('.')[0]))
print('# Query: {}'.format(stat_dict['query_count']))
print('# Question: {}'.format(sum(stat_dict['question_count'])))
print('Average query len: {:.1f}'.format(np.average(stat_dict['query_len'])))
print('Average question len: {:.1f}'.format(np.average(question_len_expanded)))
# print('Corrcoef between len: {}'.format(stat_dict['correlation']))
print('Average # of functions: {:.2f}'.format(np.average(stat_dict['function_count'])))
print('Average # of variables: {:.2f}'.format(np.average(stat_dict['variable_count'])))
print('Average # of keywords: {:.2f}'.format(np.average(stat_dict['keyword_count'])))
print('Average # of distinct keywords: {:.2f}'.format(np.average(stat_dict['distinct_keyword_count'])))
plt.show()
print(0)
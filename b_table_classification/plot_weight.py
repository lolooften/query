'''
plot_weight.py

读取模型生成的注意力权重, 并画出图.
需要的文件:
    --- data_table_test.json (完成分词的句子文本).
    --- attention_weight.txt (生成的注意力矩阵).
生成的文件:
    --- 图片
'''

import numpy as np
import matplotlib.pyplot as plt
import codecs
import json

path_texts = '../a_dataset/dataset_query_based_split/data_table_test2.json'
path_weights = 'NeuralNLP/checkpoint_dir_table/attention_weight.txt'

texts = []
weights = []
with open(path_weights, 'r') as fin:
    for line in fin:
        tmp = line.split(',')
        weights.append([float(t) for t in tmp])
weights = np.array(weights)
for line in codecs.open(path_texts, "r", 'utf-8'):
    texts.append(json.loads(line.strip("\n")))

def plot_attention(texts, weights, plot_index_list=None, n=None):
    if n is not None:
        plot_index_list = np.random.randint(0, len(texts), n)
    n_index = len(plot_index_list)
    # 确定图片网格有多少行多少列
    # nrow = int(np.ceil(np.sqrt(n_index / 2)))
    nrow = 3
    ncol = int(np.ceil(n_index / nrow))
    plt.figure(figsize=(10, 2))
    for i, plot_index in enumerate(plot_index_list):
        ax = plt.subplot(nrow, ncol, i+1)
        len_text = len(texts[plot_index]['doc_token'])
        plt.imshow(np.log10(np.array([weights[plot_index][:len_text]])), cmap='Blues')
        # plt.title('#%d' % plot_index)
        ax.set_xticks(np.arange(len_text), labels=texts[plot_index]['doc_token'][:len_text], fontproperties='SimSun', fontsize=14)
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

plot_attention(texts, weights, plot_index_list=[0, 1, 2])
print(0)

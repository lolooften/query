'''
seq2seq.py

来自 https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

最基础的 seq2seq 模型.
需要的文件:
    --- data_train.txt (训练集数据)
    --- data_test.txt (测试集数据)
    (没有验证集, 所以用测试集代替)
生成的文件:
    --- best_model.pt (训练过程中在验证集上最好的模型)
'''

import torch
import torch.nn as nn

import torchtext
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd

import seaborn as sns
from scipy import stats

import os
import numpy as np
import random
import math
import time
import pickle

import transformers

from generate_output_vocab import vocab_search_or_predict
from sql_parser import SQLParser, predict_next_vocab
from sql2nl import execute_sql

from eval_functions import levenshtein_similarity, best_levenshtein_similarity, jaccard_similarity, exact_match, result_match, valid_sql

# SEED = 10
SEED = 101
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


## custom dataset
model_nl = transformers.BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
tokenize_nl = transformers.BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')


def preprocess_nl(text):
    return tokenize_nl.tokenize(text)


def tokenize_sql(text):
    return text.split()


def get_vocab(src):
    return []


def get_next_vocab(trg):
    return []


SRC = Field(tokenize=preprocess_nl,
            sequential=True,
            use_vocab=True,
            init_token='<sos>',
            eos_token='<eos>',
            batch_first=True)

TRG = Field(tokenize=tokenize_sql,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

VOCABS = LabelField(use_vocab=False,
                    preprocessing=get_vocab,
                    batch_first=True)

NEXT_VOCAB = LabelField(use_vocab=False,
                    preprocessing=get_vocab,
                    batch_first=True)


fields = [(('src', 'vocabs'), (SRC, VOCABS)), ('trg', TRG)]


train_data, valid_data, test_data = TabularDataset.splits(
    path='a_dataset/dataset_query_based_split/',
    train='data_train.txt',
    validation='data_test.txt',
    test='data_test.txt',
    format='tsv',
    fields=fields,
)


# 建立 TRG 词汇时, 需要将 x, y, u 表中的词替换加入
# 注意, 这里的词汇需要加上引号
SRC.build_vocab([[i] for i in tokenize_nl.vocab])
TRG.build_vocab(train_data)

TRG_AUX = Field()
aux_vocab = []
for f in ['a_dataset/dataset_raw/x.txt', 'a_dataset/dataset_raw/y.txt', 'a_dataset/dataset_raw/u.txt']:
    for line in open(f, 'r', encoding='utf-8'):
        # 因为在 sql 语句中, 这些词需要加引号
        aux_vocab.append(["'" + line.rstrip().split(',')[1].lower() + "'"])
TRG_AUX.build_vocab(aux_vocab)
TRG.vocab.extend(TRG_AUX.vocab)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=lambda x: len(x.src),
    sort_within_batch=True,
)


class Encoder(nn.Module):
    def __init__(self,
                 pretrained_embeddings,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding.from_pretrained(pretrained_embeddings[:, :hid_dim], freeze=True)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len]
        # src_mask = [batch_size, 1, 1, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch_size, src_len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
            # src = [batch_size, src_len, hid_dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len, hid_dim]
        # src_dim = [batch_size, 1, 1, src_len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch_size, src_len, hid_dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch_size, src_len, hid_dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        # query = [batch_size, query_len, hid_dim]
        # key = [batch_size, key_len, hid_dim]
        # value = [batch_size, value_len, hid_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch_size, n_heads, query_len, head_dim]
        # K = [batch_size, n_heads, key_len, head_dim]
        # V = [batch_size, n_heads, value_len, head_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch_size, n_heads, query_len, head_dim]
        # K = [batch_size, n_heads, key_len, head_dim]
        # V = [batch_size, n_heads, value_len, head_dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch_size, query_len, hid_dim]

        x = self.fc_o(x)
        # x = [batch_size, query_len, hid_dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, seq_len, hid_dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch_size, seq_len, pf_dim]

        x = self.fc_2(x)
        # x = [batch_size, seq_len, hid_dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len]
        # enc_src = [batch_size, src_len, hid_dim]
        # trg_mask = [batch_size, 1, trg_len, trg_len]
        # src_mask = [batch_size, 1, 1, src_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch_size, trg_len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch_size, trg_len, hid_dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            # trg = [batch_size, trg_len, hid_dim]
            # attention = [batch_size, n_heads, trg_len, src_len]

        output = self.fc_out(trg)
        # output = [batch_size, trg_len, output_dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len, hid_dim]
        # enc_src = [batch_size, src_len, hid_dim]
        # trg_mask = [batch_size, 1, trg_len, trg_len]
        # src_mask = [batch_size, 1, 1, src_len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch_size, trg_len, hid_dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # attention = [batch_size, n_heads, trg_len, src_len]

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch_size, trg_len, hid_dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch_size, trg_len, hid_dim]

        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch_size, src_len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch_size, 1, 1, src_len]

        return src_mask
    
    def make_trg_mask(self, trg):
        # trg = [batch_size, trg_len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch_size, 1, 1, trg_len]

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch_size, 1, trg_len, trg_len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch_size, src_len]
        # trg = [batch_size, trg_len]
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch_size, 1, 1, src_len]
        # trg_mask = [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch_size, src_len, hid_dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch_size, trg_len, output_dim]
        # attention = [batch_size, n_heads, trg_len, src_len]

        return output, attention

OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = .1
DEC_DROPOUT = .1

enc = Encoder(model_nl.embeddings.word_embeddings.weight,
                HID_DIM,
                ENC_LAYERS,
                ENC_HEADS,
                ENC_PF_DIM,
                ENC_DROPOUT,
                device)

dec = Decoder(OUTPUT_DIM,
                HID_DIM,
                DEC_LAYERS,
                DEC_HEADS,
                DEC_PF_DIM,
                DEC_DROPOUT,
                device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def cout_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {cout_parameters(model):,} trainable parameters.')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)

LEARNING_RATE = .0005
GAMMA = 0.95
# LEARNING_RATE = .0003

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, GAMMA)

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

def recover_sentence(field, index_list, replace_space=True):
    # 将 batch 中的 index 列表转化为自然语言句子, 以在训练过程中判断需要哪些词汇
    # 也可以将 sql 的 index 还原为 sql 句子
    pad_index = field.vocab.stoi[field.pad_token]
    init_index = field.vocab.stoi[field.init_token]
    eos_index = field.vocab.stoi[field.eos_token]
    index_list = [i for i in index_list if i != pad_index and i != init_index and i != eos_index]
    tokens = [field.vocab.itos[i] for i in index_list]
    sentence = tokenize_nl.convert_tokens_to_string(tokens)
    # remove spaces from sentence
    if replace_space:
        sentence = sentence.replace(' ', '')
    return sentence


def get_trg_vocab_01(batch_size, trg_len_m1, output_dim, trg_vocab_limited_index):
    # 按 0-1 矩阵的形式判断哪些词在 trg_vocab_limited_index 中 (trg_next_vocab_limited_index 同理)
    # 1 是允许出现的词, 0 是不允许出现的词
    trg_vocab_01 = np.zeros([batch_size, output_dim])
    for i in range(batch_size):
        for index in trg_vocab_limited_index[i]:
            trg_vocab_01[i][index] = 1
            # trg_vocab_01 = [batch_size, output_dim]

    trg_vocab_01 = np.expand_dims(trg_vocab_01, 1)
    # trg_vocab_01 = [batch_size, 1, output_dim]

    trg_vocab_01 = np.repeat(trg_vocab_01, trg_len_m1, axis=1)
    # trg_vocab_01 = [batch_size, trg_len-1, output_dim]

    trg_vocab_01 = torch.tensor(trg_vocab_01)

    return trg_vocab_01


def custom_loss(output, trg, trg_vocab_limited_index, trg_next_vocab_limited_index, criterion, alpha, beta):
    # trg_vocab_limited_index 是全局限定词, 用 alpha 做系数
    # trg_next_vocab_limited_index 是下一个词的限定词, 用 beta 做系数
    batch_size, trg_len_m1, output_dim = output.shape

    trg_vocab_01 = get_trg_vocab_01(batch_size, trg_len_m1, output_dim, trg_vocab_limited_index)
    trg_next_vocab_01 = get_trg_vocab_01(batch_size, trg_len_m1, output_dim, trg_next_vocab_limited_index)
    # trg_vocab_limited_index = list
    # trg_vocab_01 = [batch_size, trg_len, output_dim]
    # trg_next_vocab_01 = [batch_size, trg_len, output_dim]

    # 此项的目的是让模型知道不该出现的词的 output 概率应该很低
    # 用 1 减, 将不可出现的词的系数从 0 变为 1
    vocab_loss_term = torch.sum(torch.square(torch.mul(output, 1-trg_vocab_01)))
    next_vocab_loss_term = torch.sum(torch.square(torch.mul(output, 1-trg_next_vocab_01)))

    output = output.contiguous().view(-1, output_dim)
    trg = trg[:, 1:].contiguous().view(-1) # truncate the `<sos>' token at the first of the target sentence
    # output = [batch_size * (trg_len-1), output_dim]
    # trg = [batch_size * (trg_len-1)]
    
    loss = criterion(output, trg) + \
        alpha * vocab_loss_term / batch_size / trg_len_m1 + \
        beta * next_vocab_loss_term / batch_size / trg_len_m1
    
    return loss


def train(model, iterator, optimizer, criterion, clip, alpha, beta):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):
        # 若此处报错，需修改 site-packages\transformers\tokenization_utils_base.py 第 245 行:
        #    return {key: self.data[key][slice] for key in self.data.keys()}
        # -> return {key: self.data[key][item] for key in self.data.keys()}
        src = batch.src
        trg = batch.trg
        src_nl = [recover_sentence(SRC, src_i) for src_i in src]
        trg_vocab_limited = vocab_search_or_predict(src_nl)
        trg_vocab_limited = [v + [TRG.pad_token, TRG.init_token, TRG.eos_token] for v in trg_vocab_limited]
        trg_vocab_limited_index = [[TRG.vocab.stoi[i] for i in v] for v in trg_vocab_limited]
        
        # 检查是否有模型判断应该输出的词, 却不在 TRG 词典中的
        # for v in trg_vocab_limited:
        #     for i in v:
        #         if TRG.vocab.stoi[i] == 0:
        #             print(i)
        
        # 检查是否有训练数据中正确的词, 却不在模型判断应该输出的词中的
        # for i in range(trg.shape[0]):
        #     for j in trg[i]:
        #         if j != TRG.vocab.stoi[TRG.pad_token] and j != TRG.vocab.stoi[TRG.init_token] and j != TRG.vocab.stoi[TRG.eos_token]:
        #             if j not in trg_vocab_limited_index[i]:
        #                 print(i, src_nl[i])
        #                 print(i, ' '.join([TRG.vocab.itos[k] for k in trg[i]]))
        #                 print(i, TRG.vocab.itos[j])
        #                 print([TRG.vocab.itos[k] for k in trg_vocab_limited_index[i]])
        #                 print('\n')

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1]) # truncate the `<eos>' token at the end of the target sentence
        # output = [batch_size, trg_len-1, output_dim]
        # trg = [batch_size, trg_len]

        loss = custom_loss(output, trg, trg_vocab_limited_index, trg_vocab_limited_index, criterion, alpha, beta)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
    
    scheduler.step()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, alpha, beta):
    # 与 train 基本相同
    model.eval()

    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        src_nl = [recover_sentence(SRC, src_i) for src_i in src]
        trg_vocab_limited = vocab_search_or_predict(src_nl)
        trg_vocab_limited = [v + [TRG.pad_token, TRG.init_token, TRG.eos_token] for v in trg_vocab_limited]
        trg_vocab_limited_index = [[TRG.vocab.stoi[i] for i in v] for v in trg_vocab_limited]

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1]) # truncate the `<eos>' token at the end of the target sentence
        # output = [batch_size, trg_len-1, output_dim]
        # trg = [batch_size, trg_len]

        loss = custom_loss(output, trg, trg_vocab_limited_index, trg_vocab_limited_index, criterion, alpha, beta)

        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elpased_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elpased_secs


def translate_sentence(sentence, src_field, trg_field, model, parser, device, max_len=100):

    model.eval()

    if isinstance(sentence, str):
        tokens = [token.lower() for token in tokenize_nl.tokenize(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
        sentence = tokenize_nl.convert_tokens_to_string(sentence)
        sentence = sentence.replace(' ', '')
    
    trg_vocab_limited = vocab_search_or_predict(sentence) # 这里的 sentence 是 nl 字符串而不是 token 列表
    trg_vocab_limited = [v + [TRG.pad_token, TRG.init_token, TRG.eos_token] for v in trg_vocab_limited]
    trg_vocab_limited_index = [[TRG.vocab.stoi[i] for i in v] for v in trg_vocab_limited]
    trg_vocab_01 = get_trg_vocab_01(1, 1, len(trg_field.vocab.itos), trg_vocab_limited_index)

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for _ in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # current_sql = recover_sentence(trg_field, trg_indexes, replace_space=False)
        # 请注意!
        # trg_vocab_limited 是列表的列表, 因为 vocab_search_or_predict 支持一次预测多个句子
        # [0] 是因为 trg_vocab_limited 是一个列表的列表, 但这里只有 1 句句子
        # 最后用 [] 括起来才得到 trg_next_vocab_limited 是为了适配后面的函数
        # [:-3] 是将上一步额外添加的 <pad>, <sos>, <eos> 暂时删去, 否则它会照字面拼接
        # trg_next_vocab_limited = [predict_next_vocab(parser, trg_vocab_limited[0][:-3], current_sql)]
        # trg_next_vocab_limited = [v + [TRG.pad_token, TRG.init_token, TRG.eos_token] for v in trg_next_vocab_limited]
        # trg_next_vocab_limited_index = [[TRG.vocab.stoi[i] for i in v] for v in trg_next_vocab_limited]
        # trg_next_vocab_01 = get_trg_vocab_01(1, 1, len(trg_field.vocab.itos), trg_next_vocab_limited_index)

        output = torch.mul(output, trg_vocab_01)
        # output = torch.mul(output, trg_next_vocab_01)
        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(8, 10))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='Blues')

        ax.tick_params(labelsize=10)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], rotation=0, fontproperties='SimSun')
        ax.set_yticklabels(['']+translation, rotation=0)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()
    plt.show()


### train and evaluate models
N_EPOCHS = 50
CLIP = 1
ALPHA = 1e-6
BETA = 0
parser = SQLParser._get_parser()


if __name__ == '__main__':
    # choose mode: `train' or `eval'
    # mode = 'train'
    # mode = 'eval'
    mode = 'none'

    if mode == 'train':
        best_valid_loss = float('inf')
        model_save_dir =  'result_model_question_based/1e-6/'
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, clip=CLIP, alpha=ALPHA, beta=BETA)
            valid_loss = evaluate(model, valid_iterator, criterion, alpha=ALPHA, beta=BETA)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_save_dir + 'best.pt')

            torch.save(model.state_dict(), model_save_dir + 'epoch' + str(epoch) + '.pt')

            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {scheduler.get_last_lr()[-1]:.6f}')
            # print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'  Val Loss: {valid_loss:.3f} |   Val PPL: {math.exp(valid_loss):7.3f}')

    elif mode == 'eval':
        # model_save_dir_list = ['result_model_question_based/0/', 'result_model_question_based/1e-5/', 'result_model_question_based/1e-4/', 'result_model_question_based/1e-3/', 'result_model_question_based/1e-2/']
        model_save_dir_list = ['result_model_question_based/1e-6/']
        N_EPOCHS = 1
        eval_results = {
            'loss': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
            'bleu_score': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
            'levenshtein_similarity': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
            # 'best_levenshtein_similarity': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
            'jaccard_similarity': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
            'exact_match': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
            'result_match': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
            'valid_sql': np.zeros((len(model_save_dir_list), N_EPOCHS, 2)),
        }

        # candidate_trgs 是训练的时候见到过的数据, 在 seq2seq 模型中, 很容易对训练数据产生依赖
        # candidate_trgs = [vars(datum)['trg'] for datum in train_data]

        for i, model_save_dir in enumerate(model_save_dir_list):
            for epoch in range(N_EPOCHS):
                # model.load_state_dict(torch.load(model_save_dir + 'epoch' + str(epoch) + '.pt'))
                model.load_state_dict(torch.load(model_save_dir + 'best' + '.pt'))
                data_list = [train_data, test_data]
                iterator_list = [train_iterator, test_iterator]
                for j, data in enumerate(data_list):
                    iterator = iterator_list[j]
                    translation_results_filename = model_save_dir + 'translation_results_' + str(epoch) + '_' + str(j) + '.pkl'
                    try:
                        with open(translation_results_filename, 'rb') as f:
                            srcs, trgs, pred_trgs = pickle.load(f)
                        print(i, epoch, j, 'loaded from file')
                    except FileNotFoundError:
                        srcs = []
                        trgs = []
                        pred_trgs = []
                        for k, datum in enumerate(data):
                            print(i, epoch, j, k)
                            src = vars(datum)['src']
                            trg = vars(datum)['trg']

                            pred_trg, _ = translate_sentence(src, SRC, TRG, model, parser, device)
                            # cut-off <eos> token
                            pred_trg = pred_trg[:-1]

                            srcs.append(src)
                            trgs.append(trg)
                            pred_trgs.append(pred_trg)
                        with open(translation_results_filename, 'wb') as f:
                            pickle.dump([srcs, trgs, pred_trgs], f)
                    eval_results['loss'][i][epoch][j] = evaluate(model, iterator, criterion, alpha=ALPHA, beta=BETA)
                    eval_results['bleu_score'][i][epoch][j] = torchtext.data.metrics.bleu_score(pred_trgs, [[trg] for trg in trgs])
                    eval_results['levenshtein_similarity'][i][epoch][j] = levenshtein_similarity(pred_trgs, trgs)
                    # eval_results['best_levenshtein_similarity'][i][epoch][j] = best_levenshtein_similarity(pred_trgs, candidate_trgs)
                    eval_results['jaccard_similarity'][i][epoch][j] = jaccard_similarity(pred_trgs, trgs)
                    eval_results['exact_match'][i][epoch][j] = exact_match(pred_trgs, trgs)
                    eval_results['result_match'][i][epoch][j] = result_match(pred_trgs, trgs)
                    eval_results['valid_sql'][i][epoch][j] = valid_sql(pred_trgs)
                    pickle.dump([eval_results], open('result_model_question_based' + '/eval_results_best_1e-6.pkl', 'wb'))

    else: # mode == 'none'
        model_save_dir = 'result_model_query_based/1e-3/'
        epoch = 21
        # model_save_dir = 'result_model_query_based/1e-4/'
        # epoch = 15

        # load the translation file of the test set
        # translation_results_filename = model_save_dir + 'translation_results_' + str(epoch) + '_' + str(0) + '.pkl'
        translation_results_filename = model_save_dir + 'translation_results_' + str(epoch) + '_' + str(1) + '.pkl'
        model.load_state_dict(torch.load(model_save_dir + 'epoch' + str(epoch) + '.pt'))

        with open(translation_results_filename, 'rb') as f:
            srcs, trgs, pred_trgs = pickle.load(f)

        n = len(srcs)
        eval_results_each = {
                    'NL问题长度': np.array([len(src) for src in srcs]),
                    '真实SQL语句长度': np.array([len(trg) for trg in trgs]),
                    '预测SQL语句长度': np.array([len(pred_trg) for pred_trg in pred_trgs]),
                    # 'valid_sql': valid_sql(pred_trgs, each=True),
                    'Levenshtein相似度': levenshtein_similarity(pred_trgs, trgs, each=True),
                    'Jaccard相似度': jaccard_similarity(pred_trgs, trgs, each=True),
                    '精确匹配': exact_match(pred_trgs, trgs, each=True),
                    '结果匹配': result_match(pred_trgs, trgs, each=True),
                }
        
        print_text_file = open('exact_match.txt', 'w', encoding='utf-8')
        
        ## 相关性
        df = pd.DataFrame.from_dict(eval_results_each)
        def corrdot(*args, **kwargs):
            corr_r = args[0].corr(args[1], 'pearson')
            corr_text = f"{corr_r:2.4f}"
            ax = plt.gca()
            ax.set_axis_off()
            marker_size = abs(corr_r) * 6000
            ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                    vmin=-1, vmax=1, transform=ax.transAxes)
            font_size = abs(corr_r) * 12 + 6
            ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                        ha='center', va='center', fontsize=font_size)
        def corrfunc(x, y, **kws):
            r, p = stats.pearsonr(x, y)
            p_stars = ''
            if p <= 0.05:
                p_stars = '*'
            if p <= 0.01:
                p_stars = '**'
            if p <= 0.001:
                p_stars = '***'
            ax = plt.gca()
            ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes,
                        color='black', fontsize=25)



        sns.set_theme(style='white', font_scale=.8, font='SimSun')
        g = sns.PairGrid(df, aspect=1, diag_sharey=False)
        g.map_lower(sns.regplot, lowess=True, ci=False, x_jitter=.02, y_jitter=.02,
                    scatter_kws={'s': .3, 'marker': '.'},
                    line_kws={'lw': 1,  'color': 'black'})
        g.map_diag(sns.distplot, kde_kws={'color': 'black'})
        g.map_upper(corrdot)
        g.map_upper(corrfunc)
        # plt.xlabel(fontproperties='SimSun')
        # plt.ylabel(fontproperties='SimSun')
        plt.show()






        # ## 完全匹配
        # exact_match_index_list = np.where(eval_results_each['exact_match'])[0]
        # example_idx_list = exact_match_index_list

        # ## 结果匹配但并非完全匹配
        # # exact_match_index_list = np.where(eval_results_each['exact_match'])[0]
        # # result_match_index_list = np.where(eval_results_each['result_match'])[0]
        # # example_idx_list = np.where(np.logical_and(1-eval_results_each['exact_match'],
        # #                                            eval_results_each['result_match']))[0]
        

        # ## Jaccard 相似度为 1 但没有完全匹配
        # # exact_match_index_list = np.where(eval_results_each['exact_match'])[0]
        # # jaccard_1_index_list = np.where(eval_results_each['jaccard_similarity']==1)[0]
        # # example_idx_list = list(set(jaccard_1_index_list)-set(exact_match_index_list))
        
        # ## Jaccard 和 Levenshtein 相似度均低
        # # levenshtein_low_index_list = np.where(eval_results_each['levenshtein_similarity']<.25)[0]
        # # jaccard_low_index_list = np.where(eval_results_each['jaccard_similarity']<.25)[0]
        # # example_idx_list = np.where(np.logical_and(eval_results_each['jaccard_similarity']<.25,
        # #                                            eval_results_each['levenshtein_similarity']<.25))[0]

        # ## Jaccard 相似度高但 Levenshtein相似度低
        # # levenshtein_low_index_list = np.where(eval_results_each['levenshtein_similarity']<.5)[0]
        # # jaccard_low_index_list = np.where(eval_results_each['jaccard_similarity']>.9)[0]
        # # example_idx_list = np.where(np.logical_and(np.logical_and(eval_results_each['jaccard_similarity']<.4,
        # #                                            eval_results_each['levenshtein_similarity']<.4),
        # #                                            eval_results_each['exact_match']!=1))[0]

        # ## 所有论文中提到的例子
        # # example_idx_list = [4, 422, 121, 312, 45, 12, 244]
        # # example_idx_list = [121, 312, 45, 12, 244]

        # # result_list = [None for _ in range(len(example_idx_list))]
        # # pred_result_list = [None for _ in range(len(example_idx_list))]

        # example_idx_list = [4, 181, 422]

        # for i, example_idx in enumerate(example_idx_list):
        #     # if result_list[i][0] == pred_result_list[i][0]:
        #     #     print(i, example_idx, result_list[i][0])
            
        #     # src = vars(test_data.examples[example_idx])['src']
        #     # trg = vars(test_data.examples[example_idx])['trg']
            
        #     src = srcs[example_idx]
        #     trg = trgs[example_idx]
        #     pred_trg = pred_trgs[example_idx]
        #     print(example_idx, len(pred_trg))
        #     # result_list[i] = execute_sql(' '.join(trg))
        #     # pred_result_list[i] = execute_sql(' '.join(pred_trg))
        #     # if result_list[i][0] == pred_result_list[i][0]:
        #     #     print(i, example_idx, result_list[i][0])
            
        #     translation, attention = translate_sentence(src, SRC, TRG, model, parser, device)
        #     # print(i, example_idx, result_list[i][0]==pred_result_list[i][0])
        #     print(''.join(src))
        #     print(' '.join(trg))
        #     print(' '.join(pred_trg))
        #     print('J: ' + str(eval_results_each['jaccard_similarity'][example_idx]))
        #     print('L: ' + str(eval_results_each['levenshtein_similarity'][example_idx]))

        #     print('\n')
        #     print_text_file.write(''.join(src) + '\n')
        #     for i, word in enumerate(translation):
        #         translation[i] = word.replace('t_inference_data.', 't_inf_data.')
        #         translation[i] = word.replace('inference', 'inf')
        #     display_attention(src, translation, attention)
        # print_text_file.close()
        print(0)
'''
seq2seq.py

来自 https://github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

最基础的 seq2seq 模型 (LSTM).
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

import os
import numpy as np
import random
import math
import time
import pickle

import transformers

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
    def __init__(self, pretrained_embeddings, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.tok_embedding = nn.Embedding.from_pretrained(pretrained_embeddings[:, :embedding_dim], freeze=True)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src length]
        embedded = self.dropout(self.tok_embedding(src))
        # embedded = [batch size, src length, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [batch size, src length, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [batch size, src length]
        # trg = [batch size, trg length]
        batch_size = trg.shape[0]
        trg_length = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_length, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        # input = [batch size]
        for t in range(trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[:, t] if teacher_force else top1
            # input = [batch size]
        return outputs


output_dim = len(TRG.vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    model_nl.embeddings.word_embeddings.weight,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(encoder, decoder, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {count_parameters(model):,} trainable parameters")

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
model.apply(init_weights)

LEARNING_RATE = .0001

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):
        # 若此处报错，需修改 site-packages\transformers\tokenization_utils_base.py 第 245 行:
        #    return {key: self.data[key][slice] for key in self.data.keys()}
        # -> return {key: self.data[key][item] for key in self.data.keys()}
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()

        output = model(src, trg[:, :-1], teacher_forcing_ratio) # truncate the `<eos>' token at the end of the target sentence
        # output = [batch_size, trg_len-1, output_dim]
        # trg = [batch_size, trg_len]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1) # truncate the `<sos>' token at the first of the target sentence
        # output = [batch_size * (trg_len-1), output_dim]
        # trg = [batch_size * (trg_len-1)]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    # 与 train 基本相同
    model.eval()

    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()

        output = model(src, trg[:, :-1], 0) # truncate the `<eos>' token at the end of the target sentence
        # output = [batch_size, trg_len-1, output_dim]
        # trg = [batch_size, trg_len]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1) # truncate the `<sos>' token at the first of the target sentence
        # output = [batch_size * (trg_len-1), output_dim]
        # trg = [batch_size * (trg_len-1)]

        loss = criterion(output, trg)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elpased_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elpased_secs


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=100):

    model.eval()

    if isinstance(sentence, str):
        tokens = [token.lower() for token in tokenize_nl.tokenize(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
        sentence = tokenize_nl.convert_tokens_to_string(sentence)
        sentence = sentence.replace(' ', '')
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for _ in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).to(device)

        with torch.no_grad():
            input_tensor = torch.LongTensor([trg_tensor[-1]]).to(device)
            output, hidden, cell = model.decoder(input_tensor, hidden, cell)

        pred_token = output.argmax(-1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]



### train and evaluate models
N_EPOCHS = 50
CLIP = 1
TEACHER_FORCING_RATIO = .5

if __name__ == '__main__':
    # choose mode: `train' or `eval'
    # mode = 'train'
    mode = 'eval'
    # mode = 'none'

    if mode == 'train':
        best_valid_loss = float('inf')
        model_save_dir = 'result_lstm/'
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, clip=CLIP, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_save_dir + 'best.pt')

            torch.save(model.state_dict(), model_save_dir + 'epoch' + str(epoch) + '.pt')

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'  Val Loss: {valid_loss:.3f} |   Val PPL: {math.exp(valid_loss):7.3f}')

    elif mode == 'eval':
        model_save_dir_list = ['result_lstm/']
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
                    srcs = []
                    trgs = []
                    pred_trgs = []
                    for k, datum in enumerate(data):
                        print(i, j, k)
                        src = vars(datum)['src']
                        trg = vars(datum)['trg']

                        pred_trg= translate_sentence(src, SRC, TRG, model, device)
                        # cut-off <eos> token
                        pred_trg = pred_trg[:-1]

                        
                        srcs.append(src)
                        trgs.append(trg)
                        pred_trgs.append(pred_trg)

                    pickle.dump([srcs, trgs, pred_trgs], open(model_save_dir + '/translation_results_' + str(epoch) + '_' + str(j) + '.pkl', 'wb'))
                    eval_results['loss'][i][epoch][j] = evaluate(model, iterator, criterion)
                    eval_results['bleu_score'][i][epoch][j] = torchtext.data.metrics.bleu_score(pred_trgs, [[trg] for trg in trgs])
                    eval_results['levenshtein_similarity'][i][epoch][j] = levenshtein_similarity(pred_trgs, trgs)
                    # eval_results['best_levenshtein_similarity'][i][epoch][j] = best_levenshtein_similarity(pred_trgs, candidate_trgs)
                    eval_results['jaccard_similarity'][i][epoch][j] = jaccard_similarity(pred_trgs, trgs)
                    eval_results['exact_match'][i][epoch][j] = exact_match(pred_trgs, trgs)
                    eval_results['result_match'][i][epoch][j] = result_match(pred_trgs, trgs)
                    eval_results['valid_sql'][i][epoch][j] = valid_sql(trgs)
                    pickle.dump([eval_results], open(model_save_dir + '/eval_results.pkl', 'wb'))


        ## random choose 20 and predict
        for _ in range(20):
            example_idx = random.choice(list(range(len(test_data.examples))))
            src = vars(test_data.examples[example_idx])['src']
            trg = vars(test_data.examples[example_idx])['trg']

            translation = translate_sentence(src, SRC, TRG, model, device)
            print(''.join(src))
            print(' '.join(trg))
            print(' '.join(translation[:-1]))
            print('\n')

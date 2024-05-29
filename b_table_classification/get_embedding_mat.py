'''
get_embedding_mat.py

从预训练模型中提取词嵌入矩阵并保存到本地.
需要的文件:
    --- transformers 库中的预训练模型.
生成的文件:
    --- embedding.txt (每一行格式为 `[字符] 数值向量', 按空格分隔).
'''
import transformers

tokenizer = transformers.BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
model = transformers.BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')

embedding_matrix = model.embeddings.word_embeddings.weight.detach().numpy()
vocab = tokenizer.vocab
with open('embedding.txt', 'w', encoding='utf-8') as fout:
    for i, word in enumerate(vocab):
        fout.write(word + ' ' + ' '.join([str(v) for v in embedding_matrix[i]]) + '\n')

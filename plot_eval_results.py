import pickle
import matplotlib.pyplot as plt
import numpy as np

cmap = plt.get_cmap("tab10")


NCOLS = 3

eval_results = pickle.load(open('result_model_query_based/eval_results_all.pkl', 'rb'))[0]
models = ['0', '1e-5', '1e-4', '1e-3', '1e-2']

# eval_results = pickle.load(open('result_model_question_based/eval_results_all.pkl', 'rb'))[0]
# models = ['0', '1e-5', '1e-4', '1e-3', '1e-2']

# eval_results = pickle.load(open('result_lstm/eval_results_best.pkl', 'rb'))[0]
# models = ['0']

fields = ['exact_match', 'result_match', 'valid_sql', 'jaccard_similarity', 'bleu_score', 'levenshtein_similarity']
fields_showname = ['Exact match', 'Result match', 'SQL validity', 'Jaccard similarity', 'BLEU score', 'Levenshtein score']

for i, model in enumerate(models):
    print(model)
    for f in fields:
        # show last
        # print(f'{f} {eval_results[f][i][-1][0]:.4f}, {eval_results[f][i][-1][1]:.4f}')
        # show best, in terms of best evaluation on each criterion
        print(f'{f} {max(eval_results[f][i][:, 0]):.4f}, {max(eval_results[f][i][:, 1]):.4f}')
        # show best, in terms of of best total
        # best_index = np.argmax(eval_results[f][i][:, 0] * 1650 + eval_results[f][i][:, 1] * 390)
        # print(best_index)
        # print(f'{f} {eval_results[f][i][best_index][0]:.4f}, {eval_results[f][i][best_index][1]:.4f}')
    print('\n')

n_fields = len(fields)
n_model, n_epochs, n_data = eval_results[fields[0]].shape
epochs = np.arange(n_epochs)
best_dict = {}

fig, axs = plt.subplots(ncols=NCOLS, nrows=(n_fields+NCOLS-1) // NCOLS, figsize=(15, 10))
for i, field in enumerate(fields):
    best_dict[field] = np.zeros((len(models), 2))
    row = i // NCOLS
    col = i % NCOLS
    for j, model in enumerate(models):
        best_dict[field][j] = np.max(eval_results[field][j], axis=0)
        axs[row, col].plot(epochs, eval_results[field][j][:, 0], linestyle='-', color=cmap(j), label=model + ', train')
        axs[row, col].plot(epochs, eval_results[field][j][:, 1], linestyle='--', color=cmap(j), label=model + ', test')
    axs[row, col].set_title(fields_showname[i])
    if row == (n_fields+1) // NCOLS - 1:
        axs[row, col].set_xlabel('Epoch')
    else:
        axs[row, col].tick_params('x', labelbottom=False)
    if i == NCOLS-1:
        axs[row, col].legend(ncol=2)
    
plt.tight_layout()
plt.show()
print(0)

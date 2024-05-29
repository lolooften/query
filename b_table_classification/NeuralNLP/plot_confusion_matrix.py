import matplotlib.pyplot as plt
import matplotlib
from mlxtend.plotting import plot_confusion_matrix
import pickle

eval_dir = 'eval_dir_table'

label_map, conf_mat, normal_conf_mat, _ = pickle.load(open(eval_dir+ '/' + 'test_mlcm.pkl', 'rb'))
if len(label_map) == 4:
    rearrange_index = [1, 0, 3, 2, 4]
    label_map = [list(label_map)[i] for i in rearrange_index[:-1]] + ['No label']
else:
    rearrange_index = [4, 0, 3, 2, 1, 5, 6]
    label_map = [list(label_map)[2+i] for i in rearrange_index[:-1]] + ['No label']
conf_mat = conf_mat[:, rearrange_index][rearrange_index, :]


fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=False,
                                class_names=label_map,
                                norm_colormap=matplotlib.colors.LogNorm())
fig.tight_layout()
plt.show()
print(0)
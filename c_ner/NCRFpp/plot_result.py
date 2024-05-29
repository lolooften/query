import matplotlib.pyplot as plt
import numpy as np
sentence = ["今", "天", "几", "点", "几", "分", "的", "时", "候", "，", "再", "生", "器", "压", "力", "控", "制", "第", "一", "次", "超", "过", "了", "80", "？"]
# bme_label = ['O'] * 10 + ['B-variable'] + ['M-variable'] * 5 + ['E-variable'] + ['O'] * 9
bme_label = np.array([[0] * 10 + [3] + [2] * 5 + [1] + [0] * 8])
plt.figure(figsize=(10, 2))
ax = plt.subplot(1, 1, 1)
plt.imshow(bme_label)
ax.set_xticks(np.arange(len(sentence)), labels=sentence, fontproperties='SimHei')
ax.set_yticks([])
# plt.colorbar()
plt.tight_layout()
plt.show()
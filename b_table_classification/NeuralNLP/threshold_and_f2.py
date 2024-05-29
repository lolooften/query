# !/usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import sys
import time

import torch
from torch.utils.data import DataLoader

import util
from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
from dataset.collator import ClassificationType
from dataset.collator import FastTextCollator
from evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from model.classification.drnn import DRNN
from model.classification.fasttext import FastText
from model.classification.textcnn import TextCNN
from model.classification.textvdcnn import TextVDCNN
from model.classification.textrnn import TextRNN
from model.classification.textrcnn import TextRCNN
from model.classification.transformer import Transformer
from model.classification.dpcnn import DPCNN
from model.classification.attentive_convolution import AttentiveConvNet
from model.classification.region_embedding import RegionEmbedding
from model.classification.hmcn import HMCN
from model.model_util import get_optimizer, get_hierar_relations
from util import ModeType

import numpy as np
from mlcm import mlcm
import pickle
import matplotlib.pyplot as plt

ClassificationDataset, ClassificationCollator, FastTextCollator, cEvaluator,
FastText, TextCNN, TextRNN, TextRCNN, DRNN, TextVDCNN, Transformer, DPCNN,
AttentiveConvNet, RegionEmbedding


def get_classification_model(model_name, dataset, conf):
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def eval(conf, threshold_list):
    logger = util.Logger(conf)
    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    collate_fn = globals()[collate_name](conf, len(test_dataset.label_map))
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    empty_dataset = globals()[dataset_name](conf, [])
    model = get_classification_model(model_name, empty_dataset, conf)
    optimizer = get_optimizer(conf, model)
    load_checkpoint(conf.eval.model_dir, conf, model, optimizer)
    model.eval()
    is_multi = False
    if conf.task_info.label_type == ClassificationType.MULTI_LABEL:
        is_multi = True
    predict_probs = []
    standard_labels = []
    evaluator = cEvaluator(conf.eval.dir)
    for batch in test_data_loader:
        if model_name == "HMCN":
            (global_logits, local_logits, logits) = model(batch)
        else:
            logits, _ = model(batch)
        if not is_multi:
            result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
        else:
            result = torch.sigmoid(logits).cpu().tolist()
        predict_probs.extend(result)
        standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

    recall_threshold_list = np.zeros(len(threshold_list))
    precision_threshold_list = np.zeros(len(threshold_list))
    for i, threshold in enumerate(threshold_list):
        (_, precision_list, recall_list, fscore_list, right_list,
         predict_list, standard_list) = \
        evaluator.evaluate(
            predict_probs, standard_label_ids=standard_labels, label_map=empty_dataset.label_map,
            threshold=threshold, top_k=conf.eval.top_k,
            is_flat=conf.eval.is_flat, is_multi=is_multi)
        recall = recall_list[0][cEvaluator.MICRO_AVERAGE]
        precision = precision_list[0][cEvaluator.MICRO_AVERAGE]
        recall_threshold_list[i] = recall
        precision_threshold_list[i] = precision
        
        print(threshold, recall, precision)
    return recall_threshold_list, precision_threshold_list


if __name__ == '__main__':
    # config = Config(config_file=sys.argv[1])
    config_table = Config(config_file='conf/train_table.json')
    config_time = Config(config_file='conf/train_time.json')
    threshold_list = np.arange(10, 91) * .01
    recall_list_table, precision_list_table = eval(config_table, threshold_list)
    recall_list_time, precision_list_time = eval(config_time, threshold_list)
    f1_list_table = 2/(1/recall_list_table + 1/precision_list_table)
    f1_list_time = 2/(1/recall_list_time + 1/precision_list_time)
    f2_list_table = 5/(4/recall_list_table + 1/precision_list_table)
    f2_list_time = 5/(4/recall_list_time + 1/precision_list_time)
    plt.plot(threshold_list, recall_list_table, label='子任务 1 召回率', linewidth=.55, linestyle='--', color='#1f77b4')
    plt.plot(threshold_list, precision_list_table, label='子任务 1 精确率', linewidth=.55, linestyle=':', color='#1f77b4')
    plt.plot(threshold_list, f1_list_table, label='子任务 1 F1 分数', linewidth=1.1, color='#1f77b4')
    plt.plot(threshold_list, f2_list_table, label='子任务 1 F2 分数', linewidth=2.2, color='#1f77b4')
    plt.plot(threshold_list, recall_list_time, label='子任务 2 召回率', linewidth=.55, linestyle='--', color='#ff7f0e')
    plt.plot(threshold_list, precision_list_time, label='子任务 2 精确率', linewidth=.55, linestyle=':', color='#ff7f0e')
    plt.plot(threshold_list, f1_list_time, label='子任务 2 F1 分数', linewidth=1.1, color='#ff7f0e')
    plt.plot(threshold_list, f2_list_time, label='子任务 2 F2 分数', linewidth=2.2, color='#ff7f0e')
    plt.xlabel('阈值 y_c', fontproperties='SimSun')
    plt.ylabel('分类器性能', fontproperties='SimSun')
    plt.legend(ncol=2, prop={"family":"SimSun"})
    plt.tight_layout()
    plt.show()
    # pickle.dump([threshold_list, recall_list_table, precision_list_table, recall_list_time, precision_list_time],
    #             open('threshold_and_f2.pkl', 'wb'))
    print(0)
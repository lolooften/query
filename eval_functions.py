'''
eval_functions.py

一些用于评价模型输出的函数.
函数参数: 
    --- pred_trgs: 预测的 sql 序列列表, pred_trgs = []
    --- trgs: 真实的 sql 序列列表, trgs = [trg1, trg2, ...]
    --- each: 是否按列表返回
'''

import numpy as np

from sql_parser import SQLParser
from sql2nl import execute_sql
import hashlib
from manipulate_cache import load_cache_from_file, save_cache_to_file
from pyparsing import ParseException

# 用于存储 sql 语句的执行结果
sql_cache = load_cache_from_file('sql_cache.pkl')
# 用于存储 sql 语句是否是合法语句
valid_cache = load_cache_from_file('valid_cache.pkl')

def sql_search_or_execute(sql):
    sql_hash = hashlib.sha256(sql.encode()).hexdigest()
    if sql_hash in sql_cache:
        return sql_cache[sql_hash]
    else:
        raw_result = execute_sql(sql)[0]
        raw_result_hash = hashlib.sha256(str(raw_result).encode()).hexdigest()
        sql_cache[sql_hash] = raw_result_hash
        save_cache_to_file(sql_cache, 'sql_cache.pkl')
        return raw_result_hash


# bleu 分数
# 用 torchtext.data.metrics.bleu_score 即可

def levenshtein_distance(s1, s2):
    # 拷贝自 jellyfish.levenshtein_distance, 但将检查 s1 和 s2 是否是字符串删去了
    # s1 和 s2 也可以是有序列表
    if s1 == s2:
        return 0
    rows = len(s1) + 1
    cols = len(s2) + 1

    if not s1:
        return cols - 1
    if not s2:
        return rows - 1

    prev = None
    cur = range(cols)
    for r in range(1, rows):
        prev, cur = cur, [r] + [0] * (cols - 1)
        for c in range(1, cols):
            deletion = prev[c] + 1
            insertion = cur[c - 1] + 1
            edit = prev[c - 1] + (0 if s1[r - 1] == s2[c - 1] else 1)
            cur[c] = min(edit, deletion, insertion)

    return cur[-1]

def levenshtein_similarity_single(pred_trg, trg):
    # 两个列表间的 Levenshtein 相似度
    max_len = max(len(pred_trg), len(trg))
    distance = levenshtein_distance(pred_trg, trg)
    similarity = 1 - distance / max_len
    return similarity


def levenshtein_similarity(pred_trgs, trgs, each=False):
    total_count = len(trgs)
    similarity_list = np.zeros(total_count)
    for i in range(total_count):
        pred_trg = pred_trgs[i]
        trg = trgs[i]
        similarity_list[i] = levenshtein_similarity_single(pred_trg, trg)
    if each:
        return similarity_list
    else:
        return np.average(similarity_list)


def best_levenshtein_similarity(pred_trgs, candidate_trgs, each=False):
    total_count = len(pred_trgs)
    similarity_list = np.zeros(total_count)
    total_candidate_count = len(candidate_trgs)
    for i, pred_trg in enumerate(pred_trgs):
        candidate_similarity_list = np.zeros(total_candidate_count)
        for j, candidate_trg in enumerate(candidate_trgs):
            candidate_similarity_list[j] = levenshtein_similarity_single(pred_trg, candidate_trg)
        similarity_list[i] = max(candidate_similarity_list)
    if each:
        return similarity_list
    else:
        return np.average(similarity_list)


def jaccard_similarity(pred_trgs, trgs, each=False):
    total_count = len(trgs)
    similarity_list = np.zeros(total_count)
    for i in range(total_count):
        pred_trg = pred_trgs[i]
        trg = trgs[i]
        union_vocab = set(trg).union(set(pred_trg))
        intersection_vocab = set(trg).intersection(set(pred_trg))
        similarity_list[i] = len(intersection_vocab) / len(union_vocab)
    if each:
        return similarity_list
    else:
        return np.average(similarity_list)


def exact_match(pred_trgs, trgs, each=False):
    total_count = len(trgs)
    match_list = np.zeros(total_count)
    for i in range(total_count):
        pred_trg = pred_trgs[i]
        trg = trgs[i]
        if pred_trg == trg:
            match_list[i] = 1
    if each:
        return match_list
    else:
        return np.average(match_list)


def result_match(pred_trgs, trgs, each=False):
    # 使用哈希表
    # 在评价模型的 result matching rate 时, 有很多 sql 语句需要执行, 并比较它们的返回结果
    # 在一次次模型评价中，可能会有同一 sql 被多次执行的情况
    # 通过 Python 中哈希表或者字典的方式，来判断当前语句是否被执行过
    # 如果没有则执行并保存结果 (以哈希值的形式)
    # 如果有则直接调取执行结果的哈希值并比较
    total_count = len(trgs)
    match_list = np.zeros(total_count)
    for i in range(total_count):
        pred_trg = pred_trgs[i]
        trg = trgs[i]
        pred_trg_result_hash = sql_search_or_execute(' '.join(pred_trg))
        trg_result_hash = sql_search_or_execute(' '.join(trg))
        if pred_trg_result_hash == trg_result_hash:
            match_list[i] = 1
    if each:
        return match_list
    else:
        return np.average(match_list)


def valid_sql(trgs, each=False):
    total_count = len(trgs)
    success_list = np.zeros(total_count)
    parser = SQLParser._get_parser()
    for i, trg in enumerate(trgs):
        sql = ' '.join(trg)
        sql_hash = hashlib.sha256(sql.encode()).hexdigest()
        if sql_hash in valid_cache:
            success = valid_cache[sql_hash]
        else:
            print(sql, len(sql.split(' ')))
            if len(trg) > 64:
                success = False
            else:
                success, _ = parser.runTests(sql, printResults=False)
            valid_cache[sql_hash] = success
            save_cache_to_file(valid_cache, 'valid_cache.pkl')
        if success:
            success_list[i] = 1
    if each:
        return success_list
    else:
        return np.average(success_list)


if __name__ == '__main__':
    sql = 'select count ( * ) from t_rt_info where t_rt_info.unit_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info  t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ( select t_rt_info.tag_unit from t_rt_info where t_rt_info.tag_code = ('
    parser = SQLParser._get_parser()
    success, _ = parser.runTests(sql)
    print('Success: ' + str(success))
'''
sql2nl.py

在新的数据库上执行 sql 语句, 根据根据输入的问题和返回的答案, 生成自然语言回答, 并保存结果
使用了语义依存分析进行去除引导语、否定形式改写

输入:
    --- Sql 语句
    --- 数据库
输出:
    --- 自然语言回答

需要的文件:
    --- t_rt_data.csv, t_inference_data.csv, t_model_data.csv, t_rt_info.csv (用于数据集的数据库, 位于 database 文件夹中)
    --- data_canonical.txt (所有原型的 nl 和 sql 问题对)
生成的文件:
    --- sql_execution_result_canonical.pkl (原型的执行结果, 里面包含 result_list, empty_list, none_list, error_list)
    --- sql_execution_result.txt/sql_execution_error.txt/sql_execution_none.txt/sql_execution_empty.txt (将 .pkl 文件转为 .txt 文件)

'''

from a_dataset import config
import hanlp
import mysql.connector
import datetime
import time

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
sdp = hanlp.load('SEMEVAL16_ALL_ELECTRA_SMALL_ZH')
pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

db = config.db
time_last = config.time_last

replace_list = [
    ('avg (', 'avg('),
    ('max (', 'max('),
    ('min (', 'min('),
    ('sum (', 'sum('),
    ('count (', 'count('),
    ('length (', 'length('),
    ('isnull (', 'isnull('),
    ('addtime (', 'addtime('),
    ('adddate (', 'adddate('),
    ('curdate ( )', "date('" + time_last + "')"),
    ('date (', 'date('),
    ('now ( )', "'" + time_last + "'"), # 因为导出的数据库中最新时间不是 now, 而是导出时间
    # ('now (', 'now('),
    ('timediff (', 'timediff('),
    (' :', ':'),
    (': ', ':'),
]

column_meaning = {
    # t_rt_data
    'Id': '数据编号',
    'Tag_code': '位点编码',
    'Result_value': '采集值',
    'Result_time': '采集时间',
    'Alarm_status': '报警状态',
    # t_rt_info
    'Tag_desc': '位点描述',
    'Tag_unit': '量纲',
    'Key_tag': '关键变量',
    'Unit_code': '所属装置编码',
    'LLower_limit': '低低报警阈值',
    'Lower_limit': '低报警阈值',
    'Upper_limit': '高报警阈值',
    'UUpper_limit': '高高报警阈值',
    # t_model_data
    'Inference_path': '模型推断路径',
    'Root_cause': '根原因位点编码',
    # t_inference_data
    'Rt_code': '采集数据编号',
    'Inference_time': '推断时间',
    'Inference_score': '推断结果分数',
    'Root_cause': '根原因位点编码',
    'Root_status': '根原因位点状态',
    'Inference_root_cause_id': '根原因位点编码',
    'Inference_root_cause_status': '根原因位点状态',
}


def execute_sql(sql):
    cursor = db.cursor()
    time_start = time.time()
    try:
        sql = _preprocess_sql(sql)
        cursor.execute(sql)
        raw_result = cursor.fetchall() 
        column_names = cursor.column_names

        if len(raw_result) == 0:
            result = '0 result returned.'
        elif len(raw_result) > 1:
            result = '{N} results returned.'.format(N=len(raw_result))
        else:
            result = raw_result[0][0]
    except Exception as e:
        raw_result = e
        column_names = None
        result = e
    finally:
        time_cost = time.time() - time_start
        return raw_result, column_names, result, time_cost


def _preprocess_sql(sql):
    # 将 sql 语句内不必要的空格移除
    for k, v in replace_list:
        sql = sql.replace(k, v)
    return sql


def _preprocess_question(question_nl):
    token_list = tok(question_nl)
    graph = sdp(token_list)

    # 找到句子中第一个 (id 最小) 客事关系 (Cont)/ 嵌套客事关系 (dCont)
    # 客事关系: 今天的日期 -> 查询今天的日期 / 告诉我今天的日期 / 帮我看看今天的日期
    # 嵌套客事关系: 今天是星期几 -> 查询今天是星期几 / 告诉我今天是星期几 / 帮我看看今天是星期几
    root_node_id = None
    for node in graph:
        dep_rel_list = [dep[1] for dep in node['deps']]
        # if 'Cont' in dep_rel_list or 'dCont' in dep_rel_list:
        # ### 考虑客事关系会带出很多麻烦, 暂时只考虑嵌套客事关系
        if 'dCont' in dep_rel_list:
            if root_node_id is None or root_node_id > node['id']:
                root_node_id = node['id']
    # 如果没有客事关系/嵌套客事关系, 那么所有的节点都需要保留
    if root_node_id is None:
        processed_question_nl = question_nl
    # 如果有客事关系/嵌套客事关系, 找到这个节点, 并提取由它作为根节点的所有子图
    else:
        node_id_set = {root_node_id}
        n_nodes = len(node_id_set)
        while True:
            for node in graph:
                dep_id_list = [dep[0] for dep in node['deps']]
                if any([dep_id in node_id_set for dep_id in dep_id_list]):
                    node_id_set.add(node['id'])
            if len(node_id_set) == n_nodes:
                break
            else:
                n_nodes = len(node_id_set)
        # 将所有 id 在 node_id_set 中的词连接起来, 成为新句子
        processed_question_nl = ''.join([node['form'] for node in graph 
                                        if node['id'] in node_id_set
                                        ])
    # 忽略句末的标点
    if processed_question_nl[-1] in ['。', '？', '.', '?']:
        processed_question_nl = processed_question_nl[:-1]
    return processed_question_nl


def _negative_form(nl):
    # 将句子改写为否定, 只需将根节点前添加否定词
    token_list = tok(nl)
    tag_list = pos(token_list)
    graph = sdp(token_list)

    # 找到根节点
    root_node_id = None
    for node in graph:
        dep_rel_list = [dep[1] for dep in node['deps']]
        if 'Root' in dep_rel_list:
            root_node_id = node['id']
            break
    # 判断根节点词性, 并寻找否定词
    root_node_tag = tag_list[root_node_id-1]
    # VA (表语形容词): 我 很好 -> 不
    # VC (系动词): 我 是 学生 -> 不
    # VE (动词有无): 我 有 手机 -> 没
    # VV (其他动词): 我 超过 他 -> 没
    if root_node_tag in ['VE', 'VV']:
        negative_prefix = '没'
    else:
        negative_prefix = '不'

    negative_nl = ''.join([node['form'] for node in graph[:root_node_id-1]]
                          + [negative_prefix]
                          + [node['form'] for node in graph[root_node_id-1:]])
    return negative_nl


def _generate_feature_set(question_nl):
    # 特征集: 原文献有 8 个
    # F1: (特指/选择问) 语气词, `呢'
    # F2: (是非问) 语气词, `吗、么、嘛、吧'
    # F3: (特指问) 疑问代词, `什么、如何、哪、哪里、几、谁、啥、为啥、何、何不、为何、为什么、怎么、咋、干吗、多 X'
    # F4: (是非问) 疑问格式, 能愿动词 + 语气词
    # F5: (选择问) 疑问格式, `X 还是 X'
    # F6: (正反问) 疑问格式, `X 不 X、X 不、X 没有、X 不成'
    # F7: 语气副词, `莫非、莫不是、难道、难不成、到底、何必、何须、何妨、何曾、何尝、何不、何苦、究竟、岂'
    # F8: 补充特征, F3、F5、F6 的补充特征
    # 这里选取有代表性、数据集中存在的 3 个 (F2, F3, F6)
    # 并且只做三分类, 0: 非问句; 1: 是非; 2: 特指; 3: 正反
    # 数据集中只有这三类问题, 剩下的都是陈述性语句 (非问句)
    feature_set = [0 for _ in range(3)]
    
    f2_list = ['吗', '么', '嘛', '吧']
    f2_exclusion_list = ['什么']
    if any([question_nl.endswith(w) for w in f2_list]) and not any([question_nl.endswith(w) for w in f2_exclusion_list]):
        feature_set[0] = 1
    f3_list = ['什么', '如何', '哪', '哪里', '几', '谁', '啥', '为啥', '何', '何不', '为何', '为什么', '怎么', '咋', '干嘛', '多少', '多大', '多久', '多长']
    if any([w in question_nl for w in f3_list]):
        feature_set[1] = 1
    f6_list = ['是不是', '有没有', '在不在', '存不存在', '到没到', '是否', '有无']
    f6_exclusion_list = ['所有没有', '但是不是']
    if any([w in question_nl for w in f6_list]) and not any([w in question_nl for w in f6_exclusion_list]):
        feature_set[2] = 1
    return feature_set


def _question_type_classification(question_nl):
    # 0: 非问句; 1: 是非; 2: 特指; 3: 正反
    # 原文用有限状态自动机基于 feature_set 分类 (8 个 feature, 255 种)
    feature_set = _generate_feature_set(question_nl)
    if feature_set[0] == 1:
        question_type = 1
    elif feature_set[1] == 1:
        question_type = 2
    elif feature_set[2] == 1:
        question_type = 3
    else:
        question_type = 0
    return question_type


def _from_factoid(question_nl, factoid):
    # 将 datetime 类型的 factoid 转化成自然语言
    if type(factoid) == datetime.datetime:
        factoid = factoid.strftime('%Y年%m月%d日%H时%M分%S秒')
    elif type(factoid) == datetime.timedelta:
        total_seconds = int(factoid.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        factoid = ''
        if days > 0:
            factoid += '{N}天'.format(N=int(days))
        if hours > 0:
            factoid += '{N}小时'.format(N=int(hours))
        if minutes > 0:
            factoid += '{N}分钟'.format(N=int(minutes))
        if seconds > 0:
            factoid += '{N}秒'.format(N=int(seconds))
    elif type(factoid) == datetime.date:
        factoid = factoid.strftime('%y年%m月%d日')
    # 问题分类
    question_type = _question_type_classification(question_nl)

    # token_list = tok(question_nl)
    # 非问题
    if question_type == 0:
        answer_nl = question_nl + '是' + str(factoid)
        return answer_nl
    # 是非
    if question_type == 1:
        if factoid == 0:
            # TODO 修改句子为否定形式
            answer_nl = '不是，' + _negative_form(question_nl[:-1]) # 只取到 [:-1], 因为最后一个字通常是 `吗'
        elif factoid == 1:
            answer_nl = '是的，' + question_nl[:-1]
        else:
            # 回答类型与问题类型不符, 报错
            answer_nl = '不确定，可能是语句执行错误。'
        return answer_nl
    # 特指
    if question_type == 2:
        # 先分词, 然后将疑问词例如 `什么' 替换为 factoid
        token_list = tok(question_nl)
        target_list = ['什么', '如何', '哪', '哪里', '几', '谁', '啥', '为啥', '何', '何不', '为何', '为什么', '怎么', '咋', '干嘛', '多少', '多大', '多久', '多长'] # 优先匹配长词
        for i, token in enumerate(token_list):
            if any([target in token for target in target_list]):
                if '几' in token:
                    token_list[i] = token_list[i].replace('几', str(factoid))
                elif '多少' in token:
                    token_list[i] = token_list[i].replace('多少', str(factoid))
                else:
                    token_list[i] = str(factoid)
                # 如果下一个分词结果依然包含疑问词, 需要去掉, 例如 `几点几分' 被划分为 ['几点', '几分'], 几分就可以去掉
                for j, token2 in enumerate(token_list[i+1:]):
                    if any([target in token2 for target in target_list]):
                        token_list[i+1+j] = ''
                    else:
                        break
                answer_nl = ''.join(token_list)
                return answer_nl
        answer_nl = 'not_found'
        return 'not_found'
    # 正反
    if question_type == 3:
        target_list = ['是不是', '有没有', '在不在', '存不存在', '到没到']
        for target in target_list:
            if target in question_nl:
                if factoid == 0:
                    answer_nl = question_nl.replace(target, target[1:])
                elif factoid == 1:
                    answer_nl = question_nl.replace(target, target[2:])
                else:
                    answer_nl = '不确定，可能是语句执行错误。'
                return answer_nl
        if '是否' in question_nl:
            if factoid == 0:
                answer_nl = question_nl.replace('是否', '不')
            elif factoid == 1:
                answer_nl = question_nl.replace('是否', '')
            else:
                answer_nl = '不确定，可能是语句执行错误。'
            return answer_nl
        if '有无' in question_nl:
            if factoid == 0:
                answer_nl = question_nl.replace('有无', '没有')
            elif factoid == 1:
                answer_nl = question_nl.replace('有无', '有')
            else:
                answer_nl = '不确定，可能是语句执行错误。'
            return answer_nl
        answer_nl = 'not found'
        return answer_nl
    

def generate_answer(question_nl, raw_result, column_names):
    question_nl = _preprocess_question(question_nl)
    if type(raw_result) is list:
        if len(raw_result) == 0:
            answer_nl = '没有查询到相应结果。'
        else:
            if len(column_names) == 1:
                if len(raw_result) > 10:
                    factoid = raw_result[0][0]
                    if type(factoid) == datetime.datetime:
                        factoid = factoid.strftime('%Y年%m月%d日%H时%M分%S秒')
                    else:
                        factoid = str(factoid)
                    answer_nl = '查询到{N}条结果，第一条是{example}，其余结果请在计算机查看。'.format(
                        N=len(raw_result),
                        example=factoid
                        )
                elif len(raw_result) > 1:
                    factoid = [result[0] for result in raw_result]
                    if type(factoid[0]) == datetime.datetime:
                        factoid = [f.strftime('%Y年%m月%d日%H时%M分%S秒') for f in factoid]
                    else:
                        factoid = [str(f) for f in factoid]
                    factoid = '、'.join(factoid)
                    answer_nl = '查询到{N}条结果，分别是{example}。'.format(
                        N=len(raw_result),
                        example=factoid
                        )
                else:
                    factoid = raw_result[0][0]
                    answer_nl = _from_factoid(question_nl, factoid)
            else: # 查询到的结果有多列, 无法打印
                column_names_str = '、'.join([column_meaning[column_name] for column_name in column_names])
                factoid = []
                for i, value in enumerate(raw_result[0]):
                    if value is None:
                        factoid.append('没有' + column_meaning[column_names[i]])
                    else:
                        if type(value) == datetime.datetime:
                            value = value.strftime('%Y年%m月%d日%H时%M分%S秒')
                        if column_names[i] == 'Alarm_status':
                            if value == 1:
                                value = '低报警'
                            elif value == 2:
                                value = '高报警'
                            else:
                                value = '正常'
                        elif column_names[i] == 'Inference_root_cause_status':
                            if value == 1:
                                value = '下降'
                            elif value == 2:
                                value = '上升'
                            else:
                                value = '正常'
                        elif column_names[i] == 'Tag_unit':
                            if value == 'temperature':
                                value = '摄氏度'
                        if column_names[i] == 'Key_tag':
                            if value == 0:
                                factoid.append('不是关键变量')
                            if value == 1:
                                factoid.append('是关键变量')
                        else:
                            factoid.append(column_meaning[column_names[i]] + '为' + str(value))
                factoid = '、'.join(factoid)
                if len(raw_result) > 1:
                    answer_nl = '''查询到{N}条结果，并且每条结果有{M}个域，分别为{column_names}，您可以细化您的查询要求。第一条结果的具体内容如下：{example}。'''.format(
                        N=len(raw_result),
                        M=len(column_names),
                        column_names=column_names_str,
                        example=factoid
                        )
                else:
                    answer_nl = '''查询到的结果有{M}个域，分别为{column_names}，您可以细化您的查询要求。具体内容如下：{example}。'''.format(
                        M=len(column_names),
                        column_names=column_names_str,
                        example=factoid
                        )
    else:
        result = str(raw_result)
        if result.startswith('3024'): # 超时 3024 (HY000): Query execution was interrupted, maximum statement execution time exceeded
            answer_nl = '查询超时，请换一个问题。'
        else: # 如果是自动生成的 sql, 那么不太可能会有语法错误
            answer_nl = '语法错误，错误代码{N}，错误信息{msg}，请检查语法。'.format(N=result[:4], msg=result.split(': ')[-1])
    answer_nl.strip('')
    if not answer_nl.endswith('。'):
        answer_nl += '。'
    return answer_nl

if __name__ == '__main__':

    ## 统计全部数据中各种问题类型的数量
    # question_type_count_list = [0, 0, 0, 0]
    # with open('a_dataset/dataset/data_all.txt', 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f.readlines()):
    #         nl, sql = line.strip().split('\t')
    #         nl = _preprocess_question(nl)
    #         question_type = _question_type_classification(nl)
    #         question_type_count_list[question_type] += 1
    # print(question_type_count_list)
    # print(0)


    question_nl = '上一个采样数据什么？'
    sql = "select * from t_rt_data where Result_time > addtime('2023-08-31 23:59:00', '-5:00');"
    raw_result, column_names, result, _ = execute_sql(sql)
    answer_nl = generate_answer(question_nl, raw_result, column_names)
    print(answer_nl)

    # time_start = time.time()
    # result_list = []
    # with open('a_dataset/dataset/data_all.txt', 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f.readlines()):
    #         nl, sql = line.strip().split('\t')
    #         sql = _preprocess_sql(sql)
    #         raw_result, column_names, result, time_cost = execute_sql(sql)
    #         # answer_nl = generate_answer(nl, raw_result, column_names)
    #         answer_nl = ''
    #         print(i, time_cost)
    #         result_list.append([i, nl, sql, result, answer_nl, time_cost])

    # print(time.time()-time_start)

    # with open('sql2nl_result.txt', 'w', encoding='utf-8') as f:
    #     for _result in result_list:
    #         for item in _result:
    #             f.write(str(item))
    #             f.write('\n')
    #         f.write('\n')
    # with open('sql2nl_result_timeout.txt', 'w', encoding='utf-8') as f:
    #     for _result in result_list:
    #         if type(_result[3]) is mysql.connector.DatabaseError:
    #             for item in _result:
    #                 f.write(str(item))
    #                 f.write('\n')
    #             f.write('\n')
    # with open('sql2nl_result_none.txt', 'w', encoding='utf-8') as f:
    #     for _result in result_list:
    #         if _result[3] == '0 result returned.':
    #             for item in _result:
    #                 f.write(str(item))
    #                 f.write('\n')
    #             f.write('\n')

    print(0)

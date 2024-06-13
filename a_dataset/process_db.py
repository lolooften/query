'''
process_db.py

为了生成更好的实际问题, 为 result_match_rate 铺垫基础, 需要统计数据库的信息:
    --- t_rt_data 表中变量的取值范围的分位数 (0, 25 %, 50 %, 75 %, 100 %) 分位数
    --- t_inference_data 表中出现过的变量
    --- t_model_data 表中出现过的变量

需要的文件:
    --- x.txt (提供 dcs 编码和完整编码的对应关系)
    --- t_rt_data.csv, t_inference_data.csv, t_model_data.csv, t_rt_info.csv (从服务器中下载的数据库, 位于 database_raw 文件夹中)

生成的文件:
    --- aux_vars.json (包含了辅助变量)
    --- t_rt_data.csv, t_inference_data.csv, t_model_data.csv, t_rt_info.csv (用于数据集的数据库, 位于 database 文件夹中)
'''

import pandas
import numpy as np
import json
import config
import csv

all_variables = pandas.read_csv(config.path_variable, names=['description', 'tag_code', 'dcs_code'])
aux_vars_dict = {}

# ## t_rt_data
print('t_rt_data')
db_t_rt_data = pandas.read_csv(config.path_db_raw_t_rt_data, header=0)
# 统计分位数
variables_list = list(set(db_t_rt_data['Tag_code']))
quantile_dict = {}
for v in variables_list:
    db_v = db_t_rt_data.loc[db_t_rt_data['Tag_code']==v]
    result_value_v = np.array(db_v['Result_value'])
    quantile_dict[v] = [
        np.min(result_value_v),
        np.quantile(result_value_v, .25),
        np.quantile(result_value_v, .5),
        np.quantile(result_value_v, .75),
        np.max(result_value_v),
    ]
aux_vars_dict['quantile_dict'] = quantile_dict
# 删除 Inference_root_cause_id 和 Dcs_code
db_t_rt_data = db_t_rt_data.drop(columns=['Inference_root_cause_id', 'Dcs_code'])
# 将 Alarm_status 的空值替换为 0 
db_t_rt_data['Alarm_status'] = db_t_rt_data['Alarm_status'].fillna("0")
db_t_rt_data.to_csv(config.path_db_t_rt_data, encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)


## t_inference_data
print('t_inference_data')
db_t_inference_data = pandas.read_csv(config.path_db_raw_t_inference_data, header=0)
# 统计出现过的变量
inference_variable_list = []
root_variable_list = []
for i, row in db_t_inference_data.iterrows():
    if i % 1000 == 0:
        print(i)
    # 将 Inference_root_cause_status 的 'DESC'/'low' 和 'ASC'/'high' 转变为数字形式 ('DESC'/'low': 1, 'ASC'/'high': 2)
    if row['Inference_root_cause_status'] == 'DESC' or row['Inference_root_cause_status'] == 'low':
        db_t_inference_data.at[i, 'Inference_root_cause_status'] = 1
    else:
        db_t_inference_data.at[i, 'Inference_root_cause_status'] = 2
    # 将 Inference_root_path 的顺序颠倒, 并统一格式成 t_model_data 中的格式
    inference_path = row['Inference_root_path'][1:-1].split(',')
    inference_path = [v.split("'")[1] for v in inference_path]
    inference_path.reverse()
    db_t_inference_data.at[i, 'Inference_root_path'] = ','.join(inference_path)
    # v1: 被推理的变量
    # v2: 根原因变量
    v1 = inference_path[0]
    v2 = inference_path[-1] # which is equivalent to row['Inference_root_cause_id']
    inference_variable_list.append(v1)
    root_variable_list.append(v2)
dcs_code_list = list(dict.fromkeys(inference_variable_list + root_variable_list))
tag_code_list = []
for v in dcs_code_list:
    tag_code_list.append(all_variables.loc[all_variables['dcs_code']==v]['tag_code'].iloc[0])
aux_vars_dict['t_inference_data_vars'] = tag_code_list
# 删除 Inference_path 和 Inference_path_full
db_t_inference_data = db_t_inference_data.drop(columns=['Inference_path', 'Inference_path_full'])
# 将 Inference_root_path 改名为 Inference_path
db_t_inference_data = db_t_inference_data.rename(columns={'Inference_root_path': 'Inference_path'})
# 替换 Inference_root_cause_id
for i, row in db_t_inference_data.iterrows():
    v = row['Inference_root_cause_id']
    db_t_inference_data.at[i, 'Inference_root_cause_id'] = all_variables.loc[all_variables['dcs_code']==v]['tag_code'].iloc[0]
db_t_inference_data.to_csv(config.path_db_t_inference_data, encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)


## t_model_data
print('t_model_data')
db_t_model_data = pandas.read_csv(config.path_db_raw_t_model_data, header=0, nrows=747)
# 改动的有点多 (需要对行进行修改), 所以重建一个 dataframe, 只要统计 Inference_path 即可
# 将 Inference_path 和 t_inference_data 中的 Inference_path 统一格式, 并且根据一条推理路径导出它的所有截断后路径
# 格式: '推理变量,中间变量1,中间变量2,...,根原因变量' (即 t_model_data 中的格式)
# 截断即: '推理变量', '推理变量,中间变量1', '推理变量,中间变量1,中间变量2', ..., '推理变量,中间变量1,中间变量2,...,根原因变量'
# 根据 Inference_path 的第一个变量和最后一个变量即可得到推理变量和根原因变量
inference_path_list = list(db_t_model_data['Inference_path'])
inference_path_list_extended = []
for inference_path in inference_path_list:
    code_list = inference_path.split(',')
    for i in range(len(code_list)):
        inference_path_list_extended.append(','.join(code_list[:i+1]))
inference_path_list_extended = list(dict.fromkeys(inference_path_list_extended))
inference_variable_list = [inference_path.split(',')[0] for inference_path in inference_path_list_extended]
root_variable_list = [inference_path.split(',')[-1] for inference_path in inference_path_list_extended]

delete_index = [] # 因为推理变量/根原因变量不在 t_rt_info 变量列表中, 因此删去
for i in range(len(inference_path_list_extended)):
    inference_variable = inference_variable_list[i]
    root_variable = root_variable_list[i]
    inference_variable_fetch = all_variables.loc[all_variables['dcs_code']==inference_variable]['tag_code']
    root_variable_fetch = all_variables.loc[all_variables['dcs_code']==root_variable]['tag_code']
    if len(inference_variable_fetch) > 0 and len(root_variable_fetch) > 0:
        inference_variable_list[i] = inference_variable_fetch.iloc[0]
        root_variable_list[i] = root_variable_fetch.iloc[0]
    else:
        delete_index.append(i)

inference_path_list_extended = [ele for idx, ele in enumerate(inference_path_list_extended) if idx not in delete_index]
inference_variable_list = [ele for idx, ele in enumerate(inference_variable_list) if idx not in delete_index]
root_variable_list = [ele for idx, ele in enumerate(root_variable_list) if idx not in delete_index]

t_model_data_new_dict = {'Id': list(range(1, 1+len(inference_path_list_extended))),
                         'Tag_code': inference_variable_list,
                         'Inference_path': inference_path_list_extended,
                         'Inference_root_cause_id': root_variable_list
                        }

db_t_model_data_new = pandas.DataFrame.from_dict(t_model_data_new_dict)
db_t_model_data_new.to_csv(config.path_db_t_model_data, encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)
tag_code_list = list(dict.fromkeys(inference_variable_list + root_variable_list))
aux_vars_dict['t_model_data_vars'] = tag_code_list


## t_rt_info
print('t_rt_info')
db_t_rt_info = pandas.read_csv(config.path_db_raw_t_rt_info, header=0)
db_t_rt_info = db_t_rt_info.drop(columns=['Predict_tag', 'Unit_name', 'Create_time', 'Create_user', 'Update_time', 'Update_use'])

# 补充 Tag_unit, 需要与 u.txt 保持一致
replace_dict = {
    'valve': ['阀位', '开度'],
    'temperature': ['温度', '温控', '顶温', '底温', '温降'],
    'level': ['藏量', '液位', '料位'],
    'flow': ['流量', '流控', '进料', '注氯量'],
    'density': ['密度'],
    'pressure': ['压力', '压降', '压差', '顶压', '管压', '差压'],
    'power': ['功率'],
    'concentration': ['含量'],
    'ratio': ['摩尔比'],
    'speed': ['转速']
}

for i, row in db_t_rt_info.iterrows():
    flag = 0
    tag_desc = row['Tag_desc']
    if not pandas.isna(tag_desc):
        for kw in replace_dict:
            for kw_nl in replace_dict[kw]:
                if kw_nl in tag_desc:
                    db_t_rt_info.at[i, 'Tag_unit'] = kw
                    flag = 1
                    break
            if flag == 1:
                break
    if flag == 0:
        dcs_code = row['Dcs_code']
        if 'T' in dcs_code:
            tmp = 'temperature'
        elif 'P' in dcs_code:
            tmp = 'pressure'
        elif 'F' in dcs_code:
            tmp = 'flow'
        elif 'L' in dcs_code:
            tmp = 'level'
        else:
            tmp = 'valve'
        db_t_rt_info.at[i, 'Tag_unit'] = tmp
db_t_rt_info = db_t_rt_info.drop(columns=['Dcs_code'])    
db_t_rt_info.to_csv(config.path_db_t_rt_info, encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)

json.dump(aux_vars_dict, open(config.path_aux_vars, 'w'), indent=4)
print(0)
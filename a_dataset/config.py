import mysql.connector
# 在 mysql.connector 之外, 也可以使用 MySQLdb, 有 server side cursor, 对本身执行时间短的语句可能会快一些, 对执行时间长的语句应该首先优化语句
# import MySQLdb.cursors

path_raw = 'dataset_raw/'
path_text = path_raw + 'dataset.txt.ann'
path_variable = path_raw + 'x.txt'
path_device = path_raw + 'y.txt'
path_unit = path_raw + 'u.txt'
path_aux_vars = path_raw + 'aux_vars.json'

path_processed = 'dataset/'
path_dataset = path_processed + 'dataset.json'
path_dataset_gbk = path_processed + 'dataset_gbk.json'

path_db_raw = 'database_raw/'
path_db_processed = 'database/'
path_db_raw_t_rt_data = path_db_raw + 't_rt_data.csv'
path_db_raw_t_rt_info = path_db_raw + 't_rt_info.csv'
path_db_raw_t_model_data = path_db_raw + 't_model_data.csv'
path_db_raw_t_inference_data = path_db_raw + 't_inference_data.csv'
path_db_t_rt_data = path_db_processed + 't_rt_data.csv'
path_db_t_rt_info = path_db_processed + 't_rt_info.csv'
path_db_t_model_data = path_db_processed + 't_model_data.csv'
path_db_t_inference_data = path_db_processed + 't_inference_data.csv'

db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='33899194',
    database='db',
)

# using MySQLdb
# db = MySQLdb.connect(
#     host='localhost',
#     user='root',
#     password='33899194',
#     database='db',
#     cursorclass=MySQLdb.cursors.SSCursor
# )

cursor = db.cursor()
MAX_EXECUTION_TIME_SEC = 120
cursor.execute('set session max_execution_time={N};'.format(N=int(MAX_EXECUTION_TIME_SEC*1000)))
cursor.execute("set session sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));")
cursor.execute("set global sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));")

cursor.execute('select Result_time from t_rt_data order by Id desc limit 1;')
time_last = cursor.fetchall()[0][0].strftime('%Y-%m-%d %H:%M:%S')

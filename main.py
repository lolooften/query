from seq2seq_constrained import translate_sentence, SRC, TRG, model, parser, device
import torch

from sql2nl import generate_answer, execute_sql
from d_voice import tts_aip, asr_aip, record, realtime_asr
import time
import numpy as np

model_save_dir = 'result_model_query_based/1e-5/'
model.load_state_dict(torch.load(model_save_dir + 'best' + '.pt'))

folder = 'd_voice/'
question_voice_file = folder + 'question.wav'
question_text_file = folder + 'question_test.txt'
answer_text_file = folder + 'answer.txt'
answer_voice_file = folder + 'answer.wav'

if __name__ == '__main__':
    lines = open('test.txt', 'r', encoding='utf-8').readlines()
    time_list = np.zeros((8, len(lines)))
    for i, line in enumerate(lines):
    # while True:
        ## 获取问题文字 (使用实时语音识别)
        # time0 = time.time()
        # question_text = realtime_asr.get_question_text()
        # print('NL question: ' + question_text)
        
        ## 获取问题文字，录音 -> 问题语音 -> 问题文字 (使用 STT)
        # record.audio_record(question_voice_file, 5)
        # asr_aip.stt(question_voice_file, question_text_file)
        # with open(question_text_file, 'r', encoding='utf-8') as f:
        #     question_text = f.read()
        # print('NL question: ' + question_text)

        ## 问题文字 -> SQL
        
        question_text = line[:-1]
        print('NL question: ' + question_text)
        time1 = time.time()
        sql, _ = translate_sentence(question_text, SRC, TRG, model, parser, device)
        time2 = time.time()
        sql_query = ' '.join(sql[:-1])
        print('SQL query: ' + sql_query)
        
        ## SQL -> 答案文字
        raw_result, column_names, result, _ = execute_sql(sql_query)
        time3 = time.time()
        answer_text = generate_answer(question_text, raw_result, column_names)
        time4 = time.time()
        with open(answer_text_file, 'w', encoding='utf-8') as f:
            f.write(answer_text)
        print('NL answer: ' + answer_text)

        ## 答案文字 -> 答案语音
        tts_aip.tts(answer_text_file, answer_voice_file)
        time5 = time.time()

        ## 答案语音 -> 播放
        record.audio_play(answer_voice_file)
        time6 = time.time()
        # print('Asking:      {:.3f} s'.format(time1-time0))
        print('Translation: {:.3f} s'.format(time2-time1))
        print('Execution:   {:.3f} s'.format(time3-time2))
        print('NAG:         {:.3f} s'.format(time4-time3))
        print('TTS:         {:.3f} s'.format(time5-time4))
        print('Answering:   {:.3f} s'.format(time6-time5))
        print(len(question_text), len(sql), len(answer_text))
        print('\n')
        time_list[0][i] = len(question_text)
        time_list[1][i] = len(sql)
        time_list[2][i] = len(answer_text)
        time_list[3][i] = time2-time1
        time_list[4][i] = time3-time2
        time_list[5][i] = time4-time3
        time_list[6][i] = time5-time4
        time_list[7][i] = time6-time5
    for i in range(len(lines)):
        print('{:g} & & & {:g} & {:g} & {:g} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\'.format(
            i+1, *time_list[:, i], sum(time_list[3:, i])))
    print(0)
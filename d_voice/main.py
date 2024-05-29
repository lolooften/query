import tts_aip
import asr_aip
import chat_requests
import realtime_asr
from record import audio_record

record_time_single = 4 # 4 s = 25 * 160 ms

question_voice_file = 'question.wav'
asr_result_file = 'asr_result.txt'
chat_result_file = 'chat_result.txt'
answer_voice_file = 'answer.wav'

# audio_record(question_voice_file, record_time_single)
# asr_aip.stt(question_voice_file, asr_result_file)
# asr_result = open(asr_result_file, 'r', encoding='utf-8').read()
# chat_requests.request_answer(asr_result, chat_result_file)
# tts_aip.tts(chat_result_file, answer_voice_file)


# 不断接收新的语音问题
while True:
    # 不断录音并上传识别
    while True:
        audio_record(question_voice_file, record_time_single)


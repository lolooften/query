# -*- coding: utf-8 -*-
"""
实时流式识别
需要安装websocket-client库
使用方式 python realtime_asr.py 16k-0.pcm
"""
import websocket

import threading
import time
import uuid
import json
import logging
import sys

import pyaudio
import wave
# import record
import queue

import cn2an


APP_ID = 46311961
API_KEY = 'fY8REKpdfL0Pz4dmHMAND8dl'


# if len(sys.argv) < 2:
#     # pcm_file = "16k-0.pcm"
#     # pcm_file = 'question.wav'
#     pcm_file = 'long.pcm'
# else:
#     pcm_file = sys.argv[1]

flag = 0 # 0: 持续读入音频；1: 准备在下一个 FIN 结束；2: 记录到 FIN，进程结束
TEXT_OUT_FILE = 'question_test.txt'
RECORD_OUT_FILE = 'question_test.wav'
RECORD_TIME = 4
TRIGGER = '提问'


audio_buffer = queue.Queue()

logger = logging.getLogger()

"""

1. 连接 ws_app.run_forever()
2. 连接成功后发送数据 on_open()
2.1 发送开始参数帧 send_start_params()
2.2 发送音频数据帧 send_audio()
2.3 库接收识别结果 on_message()
2.4 发送结束帧 send_finish()
3. 关闭连接 on_close()

库的报错 on_error()
"""

def audio_record(out_file, rec_time):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    #RECORD_SECONDS = 5
    #WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # print("Start recording for {} seconds...".format(str(rec_time)))

    frames = []
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * rec_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # print("Recording done...")

    # 保存音频文件
    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



def send_start_params(ws):
    """
    开始参数帧
    :param websocket.WebSocket ws:
    :return:
    """
    req = {
        "type": "START",
        "data": {
            "appid": APP_ID,  # 网页上的appid
            "appkey": API_KEY,  # 网页上的 apikey
            "dev_pid": 15372,  # 识别模型
            "cuid": '00',  # 随便填不影响使用。机器的mac或者其它唯一id，百度计算UV用。
            "sample": 16000,  # 固定参数
            "format": "pcm"  # 固定参数

        }
    }
    body = json.dumps(req)
    ws.send(body, websocket.ABNF.OPCODE_TEXT)
    logger.info("send START frame with params:" + body)


def send_audio(ws):
    """
    发送二进制音频数据，注意每个帧之间需要有间隔时间
    :param  websocket.WebSocket ws:
    :return:
    """
    global flag
    while True:
        # print(flag)
        if flag == 2:
            break
        _ = audio_buffer.get()

        chunk_ms = 160  # 160ms的录音
        chunk_len = int(16000 * 2 / 1000 * chunk_ms)
        with open(RECORD_OUT_FILE, 'rb') as f:
            pcm = f.read()

        index = 0
        total = len(pcm)
        logger.info("send_audio total={}".format(total))
        while index < total:
            start_time = time.time()
            end = index + chunk_len
            if end >= total:
                # 最后一个音频数据帧
                end = total
            body = pcm[index:end]
            logger.debug("try to send audio length {}, from bytes [{},{})".format(len(body), index, end))
            ws.send(body, websocket.ABNF.OPCODE_BINARY)
            index = end
            end_time = time.time()
            time_cost = end_time - start_time
            time.sleep(chunk_ms / 1000.0 - time_cost)  # ws.send 也有点耗时，这里没有计算
            

def send_finish(ws):
    """
    发送结束帧
    :param websocket.WebSocket ws:
    :return:
    """
    req = {
        "type": "FINISH"
    }
    body = json.dumps(req)
    ws.send(body, websocket.ABNF.OPCODE_TEXT)
    logger.info("send FINISH frame")


def send_cancel(ws):
    """
    发送取消帧
    :param websocket.WebSocket ws:
    :return:
    """
    req = {
        "type": "CANCEL"
    }
    body = json.dumps(req)
    ws.send(body, websocket.ABNF.OPCODE_TEXT)
    logger.info("send Cancel frame")


def on_open(ws):
    """
    连接后发送数据帧
    :param  websocket.WebSocket ws:
    :return:
    """

    def run(*args):
        """
        发送数据帧
        :param args:
        :return:
        """
        send_start_params(ws)
        send_audio(ws)
        send_finish(ws)
        logger.debug("thread terminating")

    def record_buffer():
        global flag
        print('Start recording...')
        while True:
            if flag == 2:
                print('Recording done.')
                break
            audio_record(out_file=RECORD_OUT_FILE, rec_time=RECORD_TIME)
            audio_buffer.put(RECORD_OUT_FILE)
    
    threading.Thread(target=record_buffer).start()
    time.sleep(RECORD_TIME)
    threading.Thread(target=run).start()
    

def on_message(ws, message):
    """
    接收服务端返回的消息
    :param ws:
    :param message: json格式，自行解析
    :return:
    """
    global flag
    m = json.loads(message)
    if m['type'] == 'FIN_TEXT' and m['err_no'] == 0:
        print(m['result'])
        if TRIGGER in m['result']:
            flag = 1
            print('Target word recognized, please state your question...')
        elif flag == 1:
            print('Recording for question done...')
            with open(TEXT_OUT_FILE, 'w') as f:
                f.write(m['result'])
                f.close()
            flag = 2
            
    logger.info("Response: " + message)


def on_error(ws, error):
    """
    库的报错，比如连接超时
    :param ws:
    :param error: json格式，自行解析
    :return:
        """
    logger.error("error: " + str(error))


def on_close(ws):
    """
    Websocket关闭
    :param websocket.WebSocket ws:
    :return:
    """
    logger.info("ws close ...")
    # ws.close()


def _process_text(text):
    return cn2an.transform(text, 'cn2an') # 将中文数字转化为阿拉伯数字a


def get_question_text():
    logging.basicConfig(filename='log.txt',
                        filemode='a',
                        format='[%(asctime)-15s] [%(funcName)s()][%(levelname)s] %(message)s'
                        )
    # logger.setLevel(logging.DEBUG)  # 调整为logging.INFO，日志会少一点
    logger.setLevel(logging.INFO)
    logger.info("begin")
    # websocket.enableTrace(True)
    uri = "ws://vop.baidu.com/realtime_asr" + "?sn=" + str(uuid.uuid1())
    logger.info("uri is "+ uri)
    ws_app = websocket.WebSocketApp(uri,
                                    on_open=on_open,  # 连接建立后的回调
                                    on_message=on_message,  # 接收消息的回调
                                    on_error=on_error,  # 库遇见错误的回调
                                    on_close=on_close)  # 关闭后的回调
    ws_app.run_forever()
    with open(TEXT_OUT_FILE, 'r') as f:
        text = f.read()
        f.close()
    print('ASR result: ' + text)
    text = _process_text(text)
    return text


if __name__ == "__main__":
    logging.basicConfig(filename='log.txt',
                        filemode='a',
                        format='[%(asctime)-15s] [%(funcName)s()][%(levelname)s] %(message)s'
                        )
    # logger.setLevel(logging.DEBUG)  # 调整为logging.INFO，日志会少一点
    logger.setLevel(logging.INFO)
    logger.info("begin")
    # websocket.enableTrace(True)
    uri = "ws://vop.baidu.com/realtime_asr" + "?sn=" + str(uuid.uuid1())
    logger.info("uri is "+ uri)
    ws_app = websocket.WebSocketApp(uri,
                                    on_open=on_open,  # 连接建立后的回调
                                    on_message=on_message,  # 接收消息的回调
                                    on_error=on_error,  # 库遇见错误的回调
                                    on_close=on_close)  # 关闭后的回调
    ws_app.run_forever()
    with open(TEXT_OUT_FILE, 'r') as f:
        text = f.read()
        f.close()
    print('Text:', text)
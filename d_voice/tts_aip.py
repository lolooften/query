from aip import AipSpeech

APP_ID = '46311961'
API_KEY = 'fY8REKpdfL0Pz4dmHMAND8dl'
SECRET_KEY = 'cm4Barwl6xswW0rHWdpCrLfLXZU3Wwx2'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 将本地文件进行语音合成
def tts(file_in, file_out):
    print('Start TTS...')
    f = open(file_in, 'r', encoding='utf-8')
    command = f.read()
    if len(command) != 0:
        word = command
    f.close()
    extension = file_out.split('.')[1]
    if extension == 'wav':
        vol = 6
    elif extension == 'pcm':
        vol = 4
    elif extension == 'mp3':
        vol = 3
    result  = client.synthesis(word, 'zh', 1, {
        'vol': vol, 'per': 1,
    })
    
# 合成错误返回 dict, 正确返回文件
    if not isinstance(result, dict):
        with open(file_out, 'wb') as f:
            f.write(result)
        f.close()
        print('TTS successful.')

if __name__ == '__main__':
    tts('chat_result.txt', 'result.wav')

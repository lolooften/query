from aip import AipSpeech

APP_ID = '46311961'
API_KEY = 'fY8REKpdfL0Pz4dmHMAND8dl'
SECRET_KEY = 'cm4Barwl6xswW0rHWdpCrLfLXZU3Wwx2'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def get_file_content(file_in):
    with open(file_in, 'rb') as fp:
        return fp.read()

# 语音识别
def stt(file_in, file_out):
    result = client.asr(get_file_content(file_in),
                        'pcm',
                        16000,
                        { 
   'dev_pid': 1536,}      # dev_pid参数表示识别的语言类型 1536表示普通话
                        )
    print(result)


    # 解析返回值，打印语音识别的结果
    if result['err_msg'] == 'success.':
        word = result['result'][0].encode('utf-8')
        if word != '':
            if word[len(word)-3:len(word)] == '，':
                print(word[0:len(word)-3])
                with open(file_out,'wb+') as f:
                    f.write(word[0:len(word)-3])
                f.close()
            else:
                print(word.decode('utf-8').encode('gbk'))
                with open(file_out, 'wb+') as f:
                    f.write(word)
                f.close()
        else:
            print("音频文件不存在或格式错误")
    else:
        print("错误")

if __name__ == '__main__':
    stt('question.wav', 'asr_result.txt')

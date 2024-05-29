import pyaudio
import wave
import soundfile

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

    print("Start Recording for {} seconds...".format(str(rec_time)))

    frames = []
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * rec_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording Done...")

    # 保存音频文件
    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def audio_play(file):
    print('Start playing...')

    # define stream chunk
    chunk = 1024

    # open a wav format music
    try:
        f = wave.open(file, 'rb')
    except wave.Error:
        # convert file from RIFF format to wav
        data, samplerate = soundfile.read(file)
        soundfile.write(file, data, samplerate)
        f = wave.open(file, 'rb')
    # instantiate PyAudio
    p = pyaudio.PyAudio()
    # open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    # read data
    data = f.readframes(chunk)

    # play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    # stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()
    print('Playing done.')

if __name__ == '__main__':
    audio_record('question.wav', 5)
    
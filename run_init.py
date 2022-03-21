import sys, os
import pyaudio as pa
import numpy as np
import pyworld as pw
import util
from resemblyzer import VoiceEncoder

all_signal = np.array([])

def rokuon(signal) :
    global all_signal
    signal /= 32767.
    all_signal = np.append(all_signal, signal)

if len(sys.argv) < 2:
    print('引数は "rokuon" または "wavfile" でお願いします')
    exit(1)

if __name__=='__main__' and sys.argv[1]=='wavfile':
    print('# # # # 入力ディレクトリ設定 # # # #')
    input_dir = None
    if len(sys.argv) >= 3:
        input_dir = sys.argv[2]
        print(input_dir)
    else :
        input_dir = input('wavファイルのあるディレクトリを指定 >> ')
    input_path = os.path.join(input_dir, './*.wav')
    embed, f0_logmean, f0_logstd = util.getConvertInfo(input_path)
    print('# # # # 保存中 # # # #')
    sp = None
    if len(sys.argv) >= 4:
        sp = f'{sys.argv[3]}.wavc.npz'
        print(sp)
    else :
        sp = f'{input("保存名 >> ")}.wavc.npz'
    np.savez_compressed(sp, embed=embed, f0_logmean=f0_logmean, f0_logstd=f0_logstd)
    print(f'保存完了: {sp}')

elif __name__=='__main__' and sys.argv[1]=='rokuon':
    sample_rate = 16_000
    input_buffer_size = 1024 * 2
    encoder = VoiceEncoder()

    audio = pa.PyAudio()

    print('# # # # 入力設定 # # # #')
    for i in range(audio.get_device_count()):
        data = audio.get_device_info_by_index(i)
        print(f'( {data["index"]} ): {data["name"]}')
    input_device_index = int(input('input device >> '))

    stream_in = audio.open(format=pa.paInt16,
                           channels=1,
                           rate=sample_rate,
                           frames_per_buffer=input_buffer_size,
                           input=True, input_device_index=input_device_index)

    print('''# # # # 読み上げてください # # # #
妖怪（ようかい）は、日本で伝承される民間信仰において、人間の理解を超える奇怪で異常な現象や、あるいはそれらを起こす、
不可思議な力を持つ非日常的・非科学的な存在のこと。妖（あやかし）または物の怪（もののけ）、魔物（まもの）とも呼ばれる。
妖怪は日本古来のアニミズムや八百万の神の思想と人間の日常生活や自然界の摂理にも深く根ざしており、
その思想が森羅万象に神の存在を見出す一方で、否定的に把握された存在や現象は妖怪になりうるという表裏一体の関係がなされてきた。(wikipediaより)
< 読み終わったら ⌘+c で停止, 多少読み間違っても問題ないです, また読む文書はなんでもいいです >''')
    
    try:
        while stream_in.is_active():
            sinput = stream_in.read(input_buffer_size, exception_on_overflow=False)
            signal = np.frombuffer(sinput, dtype='int16').astype(np.float)
            rokuon(signal)
    except KeyboardInterrupt:
        print('\nInterrupt.')
    finally: 
        stream_in.stop_stream()
        stream_in.close()
        audio.terminate()
        print('Stop streaming.')

    print('# # # # 音声解析中 # # # #')

    embed = encoder.embed_utterance(all_signal)
    wav = all_signal.astype(np.float64)
    f0, _ = pw.harvest(wav, sample_rate)
    logf0 = np.log(f0[np.nonzero(f0)])
    f0_logmean = logf0.mean()
    f0_logstd = logf0.std()

    print('# # # # 保存中 # # # #')
    sp = f'{input("保存名 >> ")}.wavc.npz'
    np.savez_compressed(sp, embed=embed, f0_logmean=f0_logmean, f0_logstd=f0_logstd)
    print(f'保存完了: {sp}')
else:
    print('引数は "rokuon" または "wavfile" でお願いします')
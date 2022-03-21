from realtime_vc import RealtimeVC
import torch
import numpy as np
import noisereduce as nr
from model import Generator
from util import getConvertInfo, world_join, world_split, logsp_norm, logsp_unnorm
import os

# # # 設定する変数 # # # # # # # # # # # # # # # #

# サンプルレート
SAMPLE_RATE = 16_000

# モデル構造
DIM_NECK = 32
DIM_EMB = 256
DIM_PRE = 512
FREQ = 32

# バッチサイズ
BATCH = 3

# 使用するモデル
MODEL_PATH = os.path.join(os.getcwd(), 'models/world_autovc_jp_step001800.pth')

# 変換元の特徴データ
SRC_DATA = os.path.join(os.getcwd(), 'datas/suzuki.wavc.npz')

# 変換元の特徴データ
TGT_DATA = os.path.join(os.getcwd(), 'datas/jvs010.wavc.npz')

# 音量を何倍にするか？
VOLUME_X = 1.5

# 入力にノイズキャンセルを使うか？
USE_FRONT_NC = True 

# 出力にノイズキャンセルを使うか？
USE_BACK_NC = False 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_synthe(model, emb_src, src_f0_logmean, src_f0_logstd, emb_tgt, tgt_f0_logmean, tgt_f0_logstd, device):
    """
    音声変換の処理
    """

    _sembs = np.array([ emb_src for _ in range(BATCH)])
    _tembs = np.array([ emb_tgt for _ in range(BATCH)])
    
    def analysis_resynthesis(signal):
        nonlocal _sembs, _tembs

        signal /= 32767.

        if USE_FRONT_NC: 
            signal = nr.reduce_noise(y=signal, sr=SAMPLE_RATE)

        wav_len = signal.shape[0]
        pad_size = (256*20-1) * ((signal.shape[0] // (256*20-1))+1) - signal.shape[0]
        signal = np.pad(signal, [0,pad_size], 'constant')

        f0, _, sp_in, ap = world_split(signal)
        sp = sp_in.reshape((-1, 64, 513))
        sp = logsp_norm(np.log(sp))

        mels = torch.from_numpy(sp.astype(np.float32)).clone()
        sembs = torch.from_numpy(_sembs.astype(np.float32)).clone()
        tembs = torch.from_numpy(_tembs.astype(np.float32)).clone()

        with torch.inference_mode():
            m = mels.to(device)
            se = sembs.to(device)
            te = tembs.to(device)
            _, mel_outputs_postnet, _ = model(m, se, te)
        
        sp_out = np.exp(logsp_unnorm(mel_outputs_postnet.to('cpu').detach().numpy().copy()))
        sp_out = sp_out.reshape((-1, 513)).astype(np.double)

        f0_out = np.exp( (np.ma.log(f0).data - src_f0_logmean ) / src_f0_logstd * tgt_f0_logstd + tgt_f0_logmean )

        wav_out = world_join(f0_out, sp_out, ap)
        wav_out = wav_out[:wav_len]

        if USE_BACK_NC: 
            signal = nr.reduce_noise(y=signal, sr=SAMPLE_RATE)
        
        wav_out *= VOLUME_X

        wav_out *= 32767.

        return wav_out

    return analysis_resynthesis

if __name__ == '__main__':

    print('# # # # 変換モデルロード中 # # # #')
    print('load:', MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator(DIM_NECK, DIM_EMB, DIM_PRE, FREQ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print('ロード完了')

    print('# # # # 特徴データロード中 # # # #')
    src = np.load(SRC_DATA)
    emb_src = src['embed']
    src_f0_logmean = src['f0_logmean']
    src_f0_logstd = src['f0_logstd']
    tgt = np.load(TGT_DATA)
    emb_tgt = tgt['embed']
    tgt_f0_logmean = tgt['f0_logmean']
    tgt_f0_logstd = tgt['f0_logstd']
    print('ロード完了')

    print('# # # # 初期化中 # # # #')
    rvc = RealtimeVC(sample_rate=SAMPLE_RATE, input_buffer_size=(256*20-1) * BATCH - 1, output_buffer_size=(256*20-1) * BATCH - 1)
    print('初期化完了')

    print('# # # # 入力出力設定 # # # #')
    for i in range(rvc.audio.get_device_count()):
        data = rvc.audio.get_device_info_by_index(i)
        print(f'( {data["index"]} ): {data["name"]}')
    input_device_index = int(input('input device >> '))
    output_device_index = int(input('output device >> '))

    print('# # # # 変換開始 # # # #')
    rvc.run(get_synthe(model, emb_src, src_f0_logmean, src_f0_logstd, emb_tgt, tgt_f0_logmean, tgt_f0_logstd, device), 
                       input_device_index=input_device_index, output_device_index=output_device_index)


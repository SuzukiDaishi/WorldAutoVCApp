
import numpy as np
import pyaudio as pa
import pyworld as pw
from typing import Callable, Union
from nptyping import NDArray

class RealtimeVC:
    """RealtimeVC

    リアルタイムボイスチェンジャーを作成するためのクラス
    
    以下参考にしたソースコード  
    https://gist.github.com/tam17aki/8e702542f5e16c0815e7ddcc6e14bbb8

    Args:
        sample_rate (int): 入力・出力のサンプルレート、デフォルトは 48000
        input_buffer_size (int): 入力バッファサイズ、デフォルトは 1024 * 8
        output_buffer_size (int): 出力バッファサイズ、デフォルトは 1024 * 2
    
    Attributes:
        sample_rate (int): 入力・出力のサンプルレート
        input_buffer_size (int): 入力バッファサイズ
        output_buffer_size (int): 出力バッファサイズ
        audio (pa.PyAudio): 音声入力・出力関連の処理

    Notes:
        macOSの場合VSCodeのターミナルで実行すると実行権限を取れないため、音が無くなります。
        macを使うときは別のターミナルを使ってね(^_−)−☆
    """

    def __init__(self, sample_rate: int = 48000, 
                 input_buffer_size: int = 1024 * 8, 
                 output_buffer_size: int = 1024 * 2):
        self.sample_rate: int = sample_rate
        self.input_buffer_size: int = input_buffer_size
        self.output_buffer_size: int = output_buffer_size
        self.audio: pa.PyAudio = pa.PyAudio()


    def run(self, 
            analysis_resynthesis_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
            is_use_print: bool = True,
            input_device_index: Union[int, None] = None,
            output_device_index: Union[int, None] = None):
        """run
        
        ボイスチェンジャーを動作させる
        マイク入力、スピーカー出力、音声分析合成処理を実行する
        (キー入力でストップ)

        Args:
            analysis_resynthesis_func (Callable[[NDArray[np.float64]], NDArray[np.float64]]): 音声分析合成処理のための関数
            is_use_print (bool): CUIで使いやすくするための文字を出力するか否か、デフォルトは True
        """
        stream_in = None
        stream_out = None
        if input_device_index is None:
            stream_in = self.audio.open(format=pa.paInt16,
                                        channels=1,
                                        rate=self.sample_rate,
                                        frames_per_buffer=self.input_buffer_size,
                                        input=True)
        else :
            stream_in = self.audio.open(format=pa.paInt16,
                                        channels=1,
                                        rate=self.sample_rate,
                                        frames_per_buffer=self.input_buffer_size,
                                        input=True, input_device_index=input_device_index)
        if output_device_index is None:
            stream_out = self.audio.open(format=pa.paInt16,
                                        channels=1,
                                        rate=self.sample_rate,
                                        frames_per_buffer=self.output_buffer_size,
                                        output=True)
        else :
            stream_out = self.audio.open(format=pa.paInt16,
                                        channels=1,
                                        rate=self.sample_rate,
                                        frames_per_buffer=self.output_buffer_size,
                                        output=True, output_device_index=output_device_index)
        if is_use_print: print('Let\'s talk!')
        try:
            while stream_in.is_active():
                sinput = stream_in.read(self.input_buffer_size, exception_on_overflow=False)
                signal = np.frombuffer(sinput, dtype='int16').astype(np.float64)
                output = analysis_resynthesis_func(signal)
                stream_out.write(output.astype(np.int16).tobytes())
        except KeyboardInterrupt:
            if is_use_print: print('\nInterrupt.')
        finally: 
            stream_in.stop_stream()
            stream_in.close()
            stream_out.stop_stream()
            stream_out.close()
            self.audio.terminate()
            if is_use_print: print('Stop streaming.')

if __name__ == '__main__':

    ## Example

    # 入力・出力のサンプルレート
    SAMPLE_RATE = 48000

    # 音声分析合成処理の関数の定義
    def analysis_resynthesis(signal):
        f0, sp, ap = pw.wav2world(signal, SAMPLE_RATE)
        modified_f0 = 1.9 * f0
        modified_sp = np.zeros_like(sp)
        sp_range = int(modified_sp.shape[1] * 0.75)
        for f in range(modified_sp.shape[1]):
            if (f < sp_range):
                if 0.75 >= 1.0: modified_sp[:, f] = sp[:, int(f / 0.75)]
                else: modified_sp[:, f] = sp[:, int(0.75 * f)]
            else: modified_sp[:, f] = sp[:, f]
        synth = pw.synthesize(modified_f0, modified_sp, ap, SAMPLE_RATE)
        return synth

    # ボイチェンの起動
    rvc = RealtimeVC(sample_rate=SAMPLE_RATE)
    
    # 変換開始
    rvc.run(analysis_resynthesis)

    

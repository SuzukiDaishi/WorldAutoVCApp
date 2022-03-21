# WorldAutoVCApp

This program is a voice changer using WorldAutoVC, a zero-shot real-time voice quality conversion model.

このプログラムは，ゼロショットリアルタイム声質変換モデルであるWorldAutoVCを用いた，ボイスチェンジャーです．

## 簡単な説明
1. まず，`run_init.py`を用いて入力話者の声の特徴を抽出します．
2. 次に，`run_init.py`を用いて変換したい人の声の特徴を抽出します．
3. `run_app.py`を用いて，実際にリアルタイム声質変換を体感しましょう．

## 準備

**動作確認済み環境**
- Intel CPU Only
- OS: macOS Big Sur
- Pyhton: 3.6.11

**install**

クローンする
```bash
git clone https://github.com/nakalab/WorldAutoVC.git
cd WorldAutoVC
```

Pythonで必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

学習済みモデルのダウンロード

```bash
gdown 1OfRQf3aBqz0PgMLrKUacxaWieVX_YG1E
mv ./world_autovc_jp_step001800.pth ./models/world_autovc_jp_step001800.pth
```
もしくは，[こちら](https://github.com/SuzukiDaishi/WorldAutoVCTrain)で学習する．

**声の特徴量を抽出する**

wavファイルで作成する場合
```bash
python run_init.py wavfile /wavファイル/のある/ディレクトリ/ 保存する名前
```

録音して作成する場合(説明に従って録音して保存してください)
```bash
python run_init.py rokuon
```

**もし失敗する場合には以下を参照**
- [macOSにpyaudioをインストールする](https://qiita.com/mayfair/items/abb59ebf503cc294a581)

## 実行
  
`run_app.py`の下記の設定(変数)を変更して実行してください．  

- SAMPLE_RATE: サンプルレートです(変更はおすすめしません)
- DIM_NECK, DIM_EMB, DIM_PRE, FREQ: 深層学習モデルの設定です(変更はおすすめしません)
- BATCH: バッチサイズです，小さいと高速化して，大きいと変換が安定します．
- MODEL_PATH: pytorchのモデルの重みのファイルのパスを指定してください．
- SRC_DATA: ボイスチェンジャーを使う人の特徴ファイル(~~~.wavc.npz)のパスを指定してください．
- TGT_DATA: 変換したい声の特徴ファイル(~~~.wavc.npz)のパスを指定してください．
- VOLUME_X: 音量を何倍するかを指定してください．
- USE_FRONT_NC: 変換前にノイズキャンセルをかけるかどうかを指定してください．
- USE_BACK_NC: 変換後にノイズキャンセルをかけるかどうかを指定してください．

```bash
python run_app.py
```

## 使用例
- [ZOOMで用いる場合(動画)](https://www.youtube.com/watch?v=S47uXC1JCVc)

## 参考文献
- [peisuke/AutoVC.pytorch](https://github.com/peisuke/AutoVC.pytorch): 参考にした実装
- [auspicious3000/autovc](https://github.com/auspicious3000/autovc): 本家実装
- [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879): 本論文
- [JVS (Japanese versatile speech) corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus): 学習済みモデルで使用したデータセット
- [WorldAutoVCTrain](https://github.com/SuzukiDaishi/WorldAutoVCTrain): 学習に用いるスクリプト
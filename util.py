import numpy as np
import pyworld as pw
import soundfile as sf
import librosa, glob
from resemblyzer import VoiceEncoder

SAMPLE_RATE = 16000
SP_MIN = -38.6925
SP_MAX = 4.3340

def load_wav(filename):
    x = librosa.load(filename, sr=SAMPLE_RATE)[0]
    return x

def save_wav(y, filename) :
    sf.write(filename, y, SAMPLE_RATE)

def logsp_norm(sp):
    return np.clip((sp - SP_MIN) / (SP_MAX - SP_MIN), 0, 1)

def logsp_unnorm(nsp):
    return nsp * (SP_MAX - SP_MIN) + SP_MIN

def world_split(wav, use_ap=True):
    wav = wav.astype(np.float64)
    f0, t = pw.harvest(wav, SAMPLE_RATE)
    sp = pw.cheaptrick(wav, f0, t, SAMPLE_RATE)
    if use_ap:
        ap = pw.d4c(wav, f0, t, SAMPLE_RATE)
        return f0, t, sp, ap
    else:
        return f0, t, sp

def world_join(f0, sp, ap) :
    return pw.synthesize(f0, sp, ap, SAMPLE_RATE)

def f0_conversion(f0, src_f0_logmean, src_f0_logstd, tgt_f0_logmean, tgt_f0_logstd):
    f0_out = np.exp( (np.ma.log(f0).data - src_f0_logmean ) / src_f0_logstd * tgt_f0_logstd + tgt_f0_logmean )
    f0_out[f0==0] = 0
    return f0_out

def getConvertInfo(wav_gpath):
    encoder = VoiceEncoder()
    wave = []
    embed = []
    for path in glob.glob(wav_gpath):
        wav = load_wav(path)
        wave.append(wav)
        embed.append(encoder.embed_utterance(wav))
    wave = np.array(wav).flatten()
    f0, _, _ = world_split(wave, use_ap=False)
    logf0 = np.log(f0[np.nonzero(f0)])
    f0_logmean = logf0.mean()
    f0_logstd = logf0.std()
    embed = np.array(embed).mean(axis=0)
    return embed, f0_logmean, f0_logstd
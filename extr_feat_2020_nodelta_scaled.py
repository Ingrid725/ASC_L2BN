import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


file_path = 'data/dcase_audio/'
csv_file = 'evaluation_setup/fold1_all.csv'
output_path = 'features/logmel128_scaled'
feature_type = 'logmel'

sr = 44100
# 音频的秒数
duration = 10
# 矩阵的行数——帧长
num_freq_bin = 128
# FFT窗口的长度
num_fft = 2048
# 帧的移动长度
hop_length = int(num_fft / 2)
# 矩阵的列数——帧数
# 等于(M-overlap)/hop_length由于没有重叠部分，所以为M/hop_length
num_time_bin = int(np.ceil(duration * sr / hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()


for i in range(len(wavpath)):
    # sound.read能把声音文件转为Numpy数组
    # sr是The sample rate of the file
    stereo, fs = sound.read(file_path + wavpath[i], stop=duration*sr)
    # 初始化一个行数为128（帧长），列数为431（分帧），通道数为1的矩阵
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    # 对logmel_data赋值
    logmel_data[:,:,0] = librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)
    # 求对数
    logmel_data = np.log(logmel_data+1e-8)

    feat_data = logmel_data
    # 对feat_data进行归一化
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    # 把数据存储起来6
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    if i%500 == 499:
        print("%i/%i samples done" %(i+1,len(wavpath)))

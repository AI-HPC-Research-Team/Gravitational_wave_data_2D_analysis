import h5py
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool
import pandas as pd
import sys

def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()
    
def Add_noise(path,data_type,data_type_noise,sample_rate):
    print(path[:-5]+data_type)
    smb = h5py.File(path, 'r')
    length = 24000
    noise = h5py.File('noise.hdf5','r')
    for index in range(len(smb[data_type])):
        process_bar(index + 1, length)
        P_signal = np.sum(np.square(smb[data_type][index]))
        P_noise = np.sum(np.square(noise[data_type_noise][index]))
        SNR = 10*np.log10(P_signal/P_noise)
        alpha = 10**((-10-SNR)/20)
        smb_noisy = noise[data_type_noise][index] + alpha*smb[data_type][index]
        divide = max(abs(smb_noisy))
        smb_noisy = smb_noisy/divide
        f, t, Sxx = signal.spectrogram(smb_noisy,fs=sample_rate)
        f = np.log10(np.delete(f, [0]))
        start = f[0] - 0.1
        f = np.insert(f, 0, start)
        plt.figure(figsize=(5, 3))
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if 'train' in data_type:
            if 11413 < index < 20000:
                plt.savefig('Image-10db/train/' + path[:-5] + '/' + path[:-5] + '_train_' + str(index) + '.JPEG')
                plt.close()
            if 20000<index<24000:
                plt.savefig('Image-10db/val/' + path[:-5] + '/' + path[:-5] + '_validate_' + str(index) + '.JPEG')
                plt.close()
            if index > 24000:
                break

if __name__ == "__main__" :
    path = ['bwd.hdf5','emri.hdf5','sgwb.hdf5','smbhb.hdf5']
    data_type = ["train_clean","train_clean","train_clean","train_clean"]
    data_type_noise = ["train_noisy","train_noisy","train_noisy","train_noisy"]
    sample_rate = [0.1,1,0.1,0.1]
    # label = [0,1,2,3,4]
    pool = Pool(processes=10)
    pool.starmap(Add_noise,zip(path,data_type,data_type_noise,sample_rate))
    pool.close()
    pool.join()

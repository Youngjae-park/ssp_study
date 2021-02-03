#-*- coding: utf-8 -*-
"""
Created on 2021. 01. 10. (Sun) 00:41:29 KST

@author: youngjae-park
"""

from LSF import encoder, decoder
import math
import numpy as np
import os

def SNR(x, x_hat):
    err = np.zeros(x.shape[0], dtype=np.float)
    err = x - x_hat
    
    """
    print((x**2).sum())
    print((err**2).sum())

    snr = 10*np.log10((x ** 2).sum() / (err ** 2).sum())
    return snr
    """

    sum_x = np.dot(x,x)
    sum_e = np.dot(err,err)

    snr = 10*math.log10(sum_x/sum_e)

    return snr

if __name__ == '__main__':
    #filename = '/home/dhodwo/ssp_study/ssp_assignment/data/timit_wav_selected/fecd0/sa1.wav'
    filepath = '/home/dhodwo/ssp_study/ssp_assignment/data/timit_wav_selected/'
    M = 10

    for folder in os.listdir(filepath):
        for wav in os.listdir(os.path.join(filepath, folder)):
            if wav.endswith('.wav'):
                filename = os.path.join(filepath,folder,wav)
                enc = encoder()
                A, E = enc.wavfilevocoder(filename, M)
                dec = decoder(enc, A, E, enc.L)
                x_hat = dec.LPdecoder()
    
                print("WAV file name: {}".format(wav))
                print("SNR: {}".format(SNR(enc.x, x_hat)))

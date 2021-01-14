#-*- coding: utf-8 -*-
"""
Created on 2021. 01. 10. (Sun) 00:41:29 KST

@author: youngjae-park
"""

from LPC import encoder, decoder
import math
import numpy as np

def SNR(x, x_hat):
    #err = np.zeros(x.shape[0])
    err = x - x_hat
        
    sum_x = np.dot(x,x)
    sum_e = np.dot(err,err)

    snr = 10*math.log10(sum_x/sum_e)

    return snr

if __name__ == '__main__':
    filename = '/home/dhodwo/ssp_study/ssp_assignment/data/timit_wav_selected/fecd0/sa1.wav'
    M = 10

    enc = encoder()
    A, E = enc.wavfilevocoder(filename, M)
    dec = decoder(enc, A, E)
    x_hat = dec.LPdecoder()

    print(SNR(enc.x, x_hat))

#-*- coding: utf-8 -*-
"""
Created on 2021. 01. 10. (Sun) 00:41:29 KST

@author: youngjae-park
"""

import numpy as np
import librosa
from scipy.signal import lfilter

class encoder:
    def __init(self):
        pass
    
    # step A: short-time framining
    def wavfilevocoder(self, wavfilename, M, Ts=0.01):
        '''
        x: spectrum data of wav_file, Fs: sampling rate(sampling frequency)
        Ns: shift sample length
        T: the length of x
        K: the ceiling number of frame
        '''
        self.x, self.Fs = librosa.load(wavfilename, sr=8000)
        self.Ns = int(self.Fs*Ts)
        self.T = self.x.shape[0]
        self.K = (self.T-self.Ns-1)//self.Ns
        self.M = M
        
        #init A, E ndarray where A is the i
        A = np.ndarray((M+1, self.K))
        E = np.ndarray((self.Ns, self.K))
        
        for k in range(self.K):
            '''
            A = [ 1 A(2) ... A(N+1) ]
            E = linear_filter(A,1,x_k) 
            more details in scipy.signal.lfilter docs
            '''
            x_k = self.x[(k*self.Ns):((k+1)*self.Ns)]
            A[:,k] = librosa.lpc(x_k, M) 
            E[:,k] = lfilter(A[:,k],1,x_k) 

        return A, E

    # step B: pre-emphasis
    def preemphasis(self, x_mem=0, alpha=0.98):
        '''
        x_mem: the memory for calculating the first sample, the last sample of previous frame - x[k-1,Ns]
        '''
        self.x_p = np.ndarray(self.x.shape)
        self.x_p[0] = self.x[0] - alpha*x_mem
        self.x_p[1:] = self.x[1:] - alpha*self.x[0:-1]
        
        return self.x_p
    
    ''' No need to deemphasis
    # step B: de-emphasis
    def deemphasis(self, alpha=0.98):
        if not self.x_p:
            print("Doesn't have preemphasis data")
            continue
        else:
            self.x_d = self.x_p + alpha*self.x_d
    '''

    #step C: tiny noise addition
    def tinynoiseadd(self, epsilon=10e-6):
        '''
        x_pn = x_p + epsilon*(gaussian random value)
        epsilon:
        10e-6 ~ 10e-10 is proper for x is [-1,1]
        10e-3 is proper for x is [-2^15, 2^15-1]
        '''
        self.x_pn = self.x_p + epsilon*np.random.randn(self.x_p.shape[0])
        return self.x_pn

    #step D: hamming window
    def hammingwindow(self, k, method='sync'):
        '''
        hamming_window: w(n) = 0.54 - 0.46cos(2*pi*n/M-1)
        x_an: analysis frame which is the frame concatenated previous 10ms frame and present 10ms frame
        x_pnw = hamming_window*x_an
        '''
        hamming_window = np.hamming(2*self.Ns)
        if method=='sync':
            if k == 0:
                x_an = np.concatenate((self.x_pn[:self.Ns], self.x_pn[:self.Ns]))
            elif k==self.K:
                x_an = np.concatenate((self.x_pn[(self.K-1)*self.Ns:(self.K)*self.Ns], self.x_pn[(self.K-1)*self.Ns:(self.K)*self.Ns]))
            elif k>self.K:
                print("Error: window frame pass over the number of total frame")
                return
            else:
                x_an = self.x_pn[((k-1)*self.Ns):((k+1)*self.Ns)]
            x_pnw = x_an*hamming_window
            return x_pnw

    #step E: LPC
    def LPC(self, x_pnw):
        '''
        a[0], ... , a[M] = lpc(wham(320)*(x_pn[k-1,:],x_pn[k,:]))
        '''
        A = librosa.lpc(x_pnw, self.M)
        
        return A

    #step F: LPCenc
    def LPCenc(self, x_pn, A, x_mem):
        '''
        # need to check a_0 = 1.0, a_0 is A[0] under line
        e[t] = A[0]*x[t] + summation i for 1 to M (A[i]*x[t-i])
        '''
        if x_mem == 0:
            x_mem = np.zeros(self.M+1)
        T = len(x_pn)
        e = np.zeros(T)
        for t in range(T):
            # the first M samples of x uses x_mem <= Q: why M?? M+1??
            # the other samples doesn't need to use x_mem, just do convolution
            if t < self.M+1:
                x_mem[:-1] = x_mem[1:]
                x_mem[-1] = x_pn[t]
                e[t] = np.dot(A, x_mem)
            else:
                e[t] = np.dot(A, x_pn[t-self.M-1:t])
        
        if False: # Modify False to True if you need to check the result
            print("T:{}\ne:{}\nA:{}".format(T,e,A))
            print("T:{}\ne.shape:{}\nA.shape:{}".format(T,e.shape,A.shape))



    

if __name__ == '__main__':
    enc = encoder()
    A, E = enc.wavfilevocoder('/home/dhodwo/ssp_study/ssp_assignment/data/timit_wav_selected/fecd0/sa1.wav', 10)
    # print(enc.x)
    print(enc.Fs)
    # print(A.shape)
    # print(E.shape)
    x_p = enc.preemphasis()
    x_pn = enc.tinynoiseadd()
    x_test = enc.hammingwindow(433)
    A = enc.LPC(x_test)
    enc.LPCenc(x_pn, A, 0)
    
    exit()
    print(x_p)
    print(x_pn)
    print("A.shape: {}\nE.shape: {}".format(A.shape,E.shape))
    print("x_pn.shape: {}".format(x_pn.shape))
    print("K: {}".format(enc.K))

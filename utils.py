#-*- coding: utf-8 -*-
"""
Created on 2021. 01. 10. (Sun) 00:41:29 KST

@author: youngjae-park
"""

import numpy as np
import librosa
from scipy.signal import lfilter

class encoder:
    def __init__(self):
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
        A = np.zeros((M+1, self.K))
        E = np.zeros((self.Ns, self.K))
        
        """
        for k in range(self.K):
            '''
            A = [ 1 A(2) ... A(N+1) ]
            E = linear_filter(A,1,x_k) 
            more details in scipy.signal.lfilter docs
            '''
            x_k = self.x[(k*self.Ns):((k+1)*self.Ns)]
            A[:,k] = librosa.lpc(x_k, M) 
            E[:,k] = lfilter(A[:,k],1,x_k)
        """
        mem_k = np.zeros(self.Ns)
        mem_pn = np.zeros(self.Ns)
        for k in range(self.K):
            x_k = self.short_time_framing(k)
            x_p = self.preemphasis(x_k, mem_k)
            x_pn = self.tinynoiseadd(x_p)
            x_pnw = self.hammingwindow(mem_pn, x_pn)
            A[:,k] = self.LPC(x_pnw)
            E[:,k] = self.LPCenc(x_pn, A[:,k], mem_pn)

            mem_k = x_k
            mem_pn = x_pn
        
        if False: # Modify False to True when you need to check the result
            print("A:{}\nE:{}".format(A,E))
            print("A.shape:{}\nE.shape:{}".format(A.shape,E.shape))

        return A, E
    
    # step A: extract short-time framing
    def short_time_framing(self, k):
        return self.x[(k*self.Ns):((k+1)*self.Ns)]

    # step B: pre-emphasis
    def preemphasis(self, x_k, x_mem, alpha=0.98):
        '''
        x_mem: the memory for calculating the first sample, the last sample of previous frame - x[k-1,Ns]
        '''
        return x_k - alpha*x_mem
    
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
    def tinynoiseadd(self, x_p, epsilon=10e-6):
        '''
        x_pn = x_p + epsilon*(gaussian random value)
        epsilon:
        10e-6 ~ 10e-10 is proper for x is [-1,1]
        10e-3 is proper for x is [-2^15, 2^15-1]
        '''
        x_pn = x_p + epsilon*np.random.randn(x_p.shape[0])
        return x_pn

    #step D: hamming window
    def hammingwindow(self, before, present, method='zero'):
        '''
        hamming_window: w(n) = 0.54 - 0.46cos(2*pi*n/M-1)
        x_an: analysis frame which is the frame concatenated previous 10ms frame and present 10ms frame
        x_pnw = hamming_window*x_an
        '''
        x_an = np.zeros(2*self.Ns)
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
        elif method == 'zero':
            x_an[:self.Ns] = before
            x_an[self.Ns:] = present
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
        T = len(x_pn)
        e = np.zeros(T)
        for t in range(T):
            # the first M samples of x uses x_mem <= Q: why M?? M+1??
            # the other samples doesn't need to use x_mem, just do convolution
            x_mem[:-1] = x_mem[1:]
            x_mem[-1] = x_pn[t]
            e[t] = np.dot(A, np.flip(x_mem[-self.M-1:]))
        
        if False: # Modify False to True if you need to check the result
            print("T:{}\ne:{}\nA:{}".format(T,e,A))
            print("T:{}\ne.shape:{}\nA.shape:{}".format(T,e.shape,A.shape))
        
        return e

class decoder:
    def __init__(self):
        pass
    
    def LPdecoer(self):
        pass
    

if __name__ == '__main__':
    enc = encoder()
    A, E = enc.wavfilevocoder('/home/dhodwo/ssp_study/ssp_assignment/data/timit_wav_selected/fecd0/sa1.wav', 10)
    #print(enc.Ns)
    # print(enc.x)
    # print(enc.Fs)
    # print(A.shape)
    # print(E.shape)
    #x_p = enc.preemphasis()
    #x_pn = enc.tinynoiseadd()
    #x_test = enc.hammingwindow(433)
    #A = enc.LPC(x_test)
    #enc.LPCenc(x_pn, A, 0)
    
    exit()
    print(x_p)
    print(x_pn)
    print("A.shape: {}\nE.shape: {}".format(A.shape,E.shape))
    print("x_pn.shape: {}".format(x_pn.shape))
    print("K: {}".format(enc.K))

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
        self.x, self.Fs = librosa.load(wavfilename, sr=16000)
        self.Ns = int(self.Fs*Ts)
        self.T = self.x.shape[0]
        self.K = (self.T-self.Ns-1)//self.Ns
        self.M = M

        self.x = self.x[:self.Ns*self.K] # cut the last lame frame
        
        #init A, E ndarray where A is the i
        A = np.zeros((M+1, self.K))
        E = np.zeros((self.Ns, self.K))
        self.x_pn = np.zeros((self.Ns, self.K))
        
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
            self.x_pn[:,k] = x_pn
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

    # step B: Pre-emphasis
    def preemphasis(self, x_k, x_mem, alpha=0.98):
        '''
        x_mem: the memory for calculating the first sample, the last sample of previous frame - x[k-1,Ns]
        x_p[t] = x[t] - alpha*x[t-1] ( when t is 0, x_mem is needed to calculate )
        '''
        T = len(x_k)
        x_p = np.zeros(T)
        x_p[0] = x_k[0] - alpha*x_mem[-1]
        x_p[1:] = x_k[1:] - alpha*x_k[:-1]

        return x_p
    
    #step C: Tiny noise addition
    def tinynoiseadd(self, x_p, epsilon=1e-6):
        '''
        x_pn = x_p + epsilon*(gaussian random value)
        epsilon:
        10e-6 ~ 10e-10 is proper for x is [-1,1]
        10e-3 is proper for x is [-2^15, 2^15-1]
        '''
        x_pn = x_p + epsilon*np.random.randn(x_p.shape[0])
        return x_pn

    #step D: Hamming window
    def hammingwindow(self, before, present, method='zero'):
        '''
        hamming_window: w(n) = 0.54 - 0.46cos(2*pi*n/M-1)
        x_an: analysis frame which is the frame concatenated previous 10ms frame and present 10ms frame
        x_pnw = hamming_window*x_an
        '''
        x_an = np.zeros(2*self.Ns)
        hamming_window = np.hamming(2*self.Ns)
        if method == 'zero':
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

class decoder(encoder):
    def __init__(self, encoder_, A_, E_):
        self.x = encoder_.x
        self.Fs = encoder_.Fs
        self.Ns = encoder_.Ns
        self.M = encoder_.M
        self.K = encoder_.K
        self.x_pn = encoder_.x_pn

        self.A = A_
        self.E = E_
    
    """
    def LPdecoder(self) :
        A = self.A.T
        E = self.E.T
        T = self.K*self.Ns

        K, Ns = E.shape
        M = A.shape[1]-1
        x_hat= np.ndarray((K,Ns))
        x_hat_pn_mem = np.zeros((M,))
        x_hat_mem=0
        
        for k in range(K) :
            # G.LPCdec
            x_hat_pn = self.LPCdec(E[k,:],A[k,:], x_hat_pn_mem)
            
            # H.de-emphasis
            x_hat[k,:]= self.deemphasis(x_hat_pn, x_hat_mem)
            
            
            x_hat_pn_mem = x_hat_pn[-M:]
            x_hat_mem = x_hat[k,-1]
            
        x_hat = x_hat.reshape([-1])
        return x_hat[:T]

    # G.LPCdec
    def LPCdec(self, e,a,mem) :
        T = len(e)
        M = len(a)-1
        x_hat_pn = np.ndarray((T,))
        x_hat_pn = np.concatenate([mem,x_hat_pn])
        for t in range(T) :
             x_hat_pn[M+t] = (e[t] - np.sum(x_hat_pn[t:t+M]*a[:0:-1]))/a[0]
        return x_hat_pn[M:]
    
    # H.de-emphasis
    def deemphasis(self, x_hat_pn, x_hat_mem=0, alpha=0.98) :
        x_hat = np.ndarray(x_hat_pn.shape)
        x_hat[0] = x_hat_pn[0] + alpha*x_hat_mem
        for t in range(1,len(x_hat)) :
            x_hat[t] = x_hat_pn[t] + (alpha * x_hat[t-1])
        return x_hat
    """
    #"""
    def LPdecoder(self):
        x_concat_hat_d = np.zeros(self.Ns*self.K)
        
        mem_pn = np.zeros(self.M, dtype=np.float)
        mem_d = np.zeros(self.Ns, dtype=np.float)
        for k in range(self.K):
            x_hat_pn = self.LPdec(self.E[:,k], self.A[:,k], mem_pn)
            x_hat_d = self.deemphasis(x_hat_pn, mem_d)

            x_concat_hat_d[k*self.Ns:(k+1)*self.Ns] = x_hat_d
            
            mem_pn = x_hat_pn[-self.M:]
            mem_d = x_hat_d
        
        return x_concat_hat_d

    # step G: LPdec
    def LPdec(self, e, a, mem):
        '''
        e[t] = a[t]*x[t] <=> e[t] = a_0x[t] + a_1x[t-1] + a_2x[t-2] + ... + a_Mx[t-M]
        => x_hat_pn[k,t] = e[k,t] - a[k,1]x_hat_pn[t-1] - a[k,2]x_hat_pn[t-2] - ... - a[k,M]x_hat_pn[t-M]
        
        * M samples of x_hat_pn memory are needed
        '''
        x_hat_pn = np.zeros(self.Ns, dtype=np.float)
        x_hat_pn = np.concatenate([mem,x_hat_pn])
        for t in range(self.Ns):
            x_hat_pn[self.M+t] = e[t] - np.dot(a[1:], np.flip(x_hat_pn[t:t+self.M]))
        
        return x_hat_pn[self.M:]
    
    # step H: De-emphasis
    def deemphasis(self, x_hat_pn, x_mem, alpha=0.98):
        '''
        x_hat_d[t] = x_hat_pn[t] + alpha*x_hat_pn[t-1]
        when t is 0, x_mem is need to caculate.
        '''
        x_hat_d = np.zeros(len(x_hat_pn), dtype=np.float)
        x_hat_d[0] = x_hat_pn[0] + alpha*x_mem[-1]
        for t in range(1, len(x_hat_d)):
            x_hat_d[t] = x_hat_pn[t] + (alpha*x_hat_d[t-1])

        return x_hat_d
    #"""

if __name__ == '__main__':
    '''
    enc = encoder()
    A, E = enc.wavfilevocoder('/home/dhodwo/ssp_study/ssp_assignment/data/timit_wav_selected/fecd0/sa1.wav', 10)
    dec = decoder(enc, A, E)
    x_hat = dec.LPdecoder()
    '''
    
    # print(enc.x.shape)
    # print(x_hat.shape)

    # print("A:{}\nE:{}\nA.shape:{}\tE.shape:{}".format(A,E,A.shape,E.shape))
    # print(enc.Ns)
    # print(enc.x)
    # print(enc.Fs)
    # print(A.shape)
    # print(E.shape)
    # x_p = enc.preemphasis()
    # x_pn = enc.tinynoiseadd()
    # x_test = enc.hammingwindow(433)
    # A = enc.LPC(x_test)
    # enc.LPCenc(x_pn, A, 0)
    
    exit()
    print(x_p)
    print(x_pn)
    print("A.shape: {}\nE.shape: {}".format(A.shape,E.shape))
    print("x_pn.shape: {}".format(x_pn.shape))
    print("K: {}".format(enc.K))

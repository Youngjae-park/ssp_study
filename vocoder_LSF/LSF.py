#-*- coding: utf-8 -*-
"""
Created on 2021. 01. 10. (Sun) 00:41:29 KST

@author: youngjae-park
"""

import numpy as np
import librosa
from scipy.signal import lfilter, deconvolve

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
        self.e = np.zeros(self.Ns*self.K)
        
        #init A, E ndarray where A is the i
        A = np.zeros((M+1, self.K))
        E = np.zeros((self.Ns, self.K))
        self.x_pn = np.zeros((self.Ns, self.K))
        
        ########################################
        
        ########################################

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

            self.e[self.Ns*k:self.Ns*(k+1)] = E[:,k]

            mem_k = x_k
            mem_pn = x_pn
        
        if False: # Modify False to True when you need to check the result
            print("A:{}\nE:{}".format(A,E))
            print("A.shape:{}\nE.shape:{}".format(A.shape,E.shape))
        
        # Make lsf from polynomial A to L(length:M)
        L = self.poly2lsf(A)

        self.A = A
        self.E = E
        self.L = L

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
    
    #step F-2: LPCtoLSF
    def poly2lsf(self, A):
        """ Prediction polynomial to line spectral frequencies.

        converts the prediction polynomial specified by A,
        into the corresponding line spectral frequencies, LSF.
        normalizes the prediction polynomial by A(1)
        
        # Line spectral frequencies are not defined for complex polynomials.
        """
        
        # Normalize the polynomial
        LSF = np.zeros((A.shape[0]-1,A.shape[1]))
        for k in range(A.shape[1]):
            if A[0,k] != 1:
                A[:,k]/=A[0]
            
            if max(np.abs(np.roots(A[:,k]))) >= 1.0:
                error('The polynomial must have all roots inside of the unit circle.')

            # Form the sum and difference filters
            p = A.shape[0] - 1  # The leading one in the polynomial is not used
            a1 = np.concatenate((A[:,k], np.array([0])))
            a2 = a1[-1::-1]
            P1 = a1 - a2    # Difference filter
            Q1 = a1 + a2    # Sum filter

            # If order is even, remove the know root at z = 1 for P1 and z = -1 for Q1
            # If odd, remove both the roots from P1

            if p%2:
                P, r = deconvolve(P1, [1,0,-1])
                Q = Q1
            else:
                P, r = deconvolve(P1, [1,-1])
                Q, r = deconvolve(Q1, [1,1])

            rP = np.roots(P)
            rQ = np.roots(Q)

            aP = np.angle(rP[1::2]) # start from [1]idx, skip space by space. ex)1,3,5,...
            aQ = np.angle(rQ[1::2])

            lsf = sorted(np.concatenate((-aP,-aQ)))
            LSF[:,k] = lsf
        
        return LSF


class decoder(encoder):
    def __init__(self, encoder_, A_, E_, L_):
        self.x = encoder_.x
        self.Fs = encoder_.Fs
        self.Ns = encoder_.Ns
        self.M = encoder_.M
        self.K = encoder_.K
        self.x_pn = encoder_.x_pn

        self.A = A_
        self.E = E_
        self.L = L_
    
    #"""
    def LPdecoder(self):
        x_concat_hat_d = np.zeros(self.Ns*self.K)
        
        mem_pn = np.zeros(self.M, dtype=np.float)
        mem_d = np.zeros(self.Ns, dtype=np.float)
        
        # flag to use lsf2lpc A
        flag_LA = True

        # function lsf2poly
        dec_A = self.lsf2poly()

        for k in range(self.K):
            if flag_LA:
                x_hat_pn = self.LPdec(self.E[:,k], self.A[:,k], mem_pn)
            else:
                x_hat_pn = self.LPdec(self.E[:,k], self.A[:,k], mem_pn)
            x_hat_d = self.deemphasis(x_hat_pn, mem_d)

            x_concat_hat_d[k*self.Ns:(k+1)*self.Ns] = x_hat_d
            
            mem_pn = x_hat_pn[-self.M:]
            mem_d = x_hat_d

        self.decoded_x = x_concat_hat_d
        
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
    
    # step G-1: LSF to LPC
    def lsf2poly(self):
        lsf = np.array(self.L)
        
        for k in range(self.L.shape[1]):
            if max(lsf[:,k]) > np.pi or min(lsf[:,k]) < 0:
                raise ValueError('Line spectral frequencies must be between 0 and pi.')

            p = lsf.shape[0]

            # Form zeros using the LSFs and unit amplitudes
            z = np.exp(1.j * lsf[:,k])
            
            # Separate the zeros to those belonging to P and Q
            rQ = z[0::2]
            rP = z[1::2]
            
            # Include the conjugates as well
            rQ = np.concatenate((rQ, np.conjugate(rQ)))
            rP = np.concatenate((rP, np.conjugate(rP)))

            # Form the polynomials P and Q, note that these should be real
            Q = np.poly(rQ)
            P = np.poly(rP)

            # Form the sum and difference filters by including known roots at z = 1
            # and z = -1

            if p%2:
                # Odd order: z= +1 and z = -1 are roots of the difference filter, P1(z)
                P1 = np.convolve(P, [1,0,-1])
                Q1 = Q
            else:
                # Even order: z = -1 is a root of the sum filter, Q1(z) and z = 1 is a 
                # root of the difference filter, P1(z)
                P1 = np.convolve(P, [1,-1])
                Q1 = np.convolve(Q, [1,1])

            # Prediction polynomial is formed by averaging P1 and Q1

            a = .5 * (P1+Q1)

        return a[0:-1:1]

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

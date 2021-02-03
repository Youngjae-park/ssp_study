from matplotlib import pyplot as plt
import numpy as np

class vocoder_plot:
    def __init__(self, N, name, encoder, decoder):
        self.N = N
        self.fig, self.axes = plt.subplots(nrows=N)
        self.axes[0].set_title(name)
        self.xticks = np.arange(encoder.T)/encoder.Fs
        sigmax = np.maximum(np.max(abs(encoder.x)), np.max(abs(decoder.decoded_x)))
        self.sigmax = np.minimum(2**15, 1.2*sigmax)
        
        self.idx = 0
        self.enc = encoder
        self.dec = decoder

        self.cmap = plt.cm.bone

    def draw_sig_res(self):
        self.axes[self.idx].plot(self.xticks, self.enc.x, label='input signal')
        self.axes[self.idx].plot(self.xticks, self.enc.e , label='residual signal')
        self.axes[self.idx].legend()
        self.axes[self.idx].axis([self.xticks[0], self.xticks[-1], -self.sigmax, self.sigmax])
        self.axes[self.idx].set_xlabel('time (seconds)')
        self.axes[self.idx].set_ylabel('value')
        
        if self.idx != self.N:
            self.idx += 1

    def draw_spect_input(self):
        spec, freq, t, cax = self.axes[self.idx].specgram(
                self.enc.x,
                Fs=self.enc.Fs,
                window=np.hamming(self.enc.Ns*2),
                NFFT=self.enc.Ns*2,
                noverlap=80,
                scale_by_freq=True,
                mode='psd',
                scale='dB',
                cmap=self.cmap)

        lab = 'input signal, PSD %.1f+/-%.1f' % (spec.mean(), spec.std())
        print(lab)
        self.axes[self.idx].text(self.enc.T/self.enc.Fs*0.05, self.enc.Fs/8, lab)
        self.axes[self.idx].set_xlabel('time (seconds)')
        self.axes[self.idx].set_ylabel('frequency (Hz)')

        if self.idx != self.N:
            self.idx += 1

    def draw_spect_LPcoeff(self):
        """

        """

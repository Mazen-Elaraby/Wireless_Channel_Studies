import numpy as np
import matplotlib, matplotlib.pyplot as plt

def plot_CIR(self):
    plt.figure()

    plt.stem(self.CIR[0,:])

    plt.title('Channel Impulse Response')
    plt.xlabel('Delay (s)')
    plt.ylabel('Magnitude (mW)')
    plt.show()

def plot_ch_env(self):
    plt.figure()
    for i in range(self.Ch_env.shape[1]):
        plt.plot(self.t, 20*np.log10(np.abs(self.Ch_env[:,i])).flatten())
    
    plt.title('Magnitude of Complex Channel Coefficients')
    plt.xlabel('time (s)')
    plt.ylabel('Magnitude') 
    plt.show()

def plot_freq_res(self):
    plt.figure()
    Nfft = int(2**np.ceil(np.log2(len(self.CIR[0,:])))) if len(self.CIR[0,:]) > 1 else 8
    freq_domain = (self.Fs/Nfft) * np.arange((-Nfft/2), (Nfft/2))
    temp = self.CIR[0,:] #hold non-zero elements of the CIR
    freq_res = np.fft.fftshift(np.fft.fft(temp[temp != 0],Nfft)) 

    plt.semilogy(freq_domain,np.abs(freq_res))
    plt.title('Channel Frequency Response')
    plt.xlabel('Frequenct (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

def plot_doppler_spectrum(self):
    plt.figure()

    Nfft =int(2**np.ceil(np.log2(self.num_samples)))
    freq_domain = (self.Fs/Nfft) * np.arange((-Nfft/2), (Nfft/2))
    doppler_spectrum = np.fft.fftshift(np.fft.fft(self.Ch_env[:,0].real,Nfft))

    plt.plot(freq_domain,np.abs(doppler_spectrum))
    plt.xlim(-self.Fd_max-50,self.Fd_max+50)
    plt.title('Doppler Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

def plot_env_hist(self):
    plt.figure()
    for i, process in enumerate(self.Ch_env.T):
        plt.hist(np.abs(process).flatten(), bins=100, density=True, label=f'Path {i}')
    
    plt.title('Histogram of Magnitude of Complex Channel Coefficients')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.show()
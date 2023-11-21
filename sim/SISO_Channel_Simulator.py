import numpy as np
from scipy.special import erfinv
from plotting_utils import *

class SISO_Channel_Simulator:
    '''
    This class provides a Single-Input-Single-Output Channel Simulator
    The Channel is a Wireless Multipath Fading Channel
    The following code is able to simulate flat & frequency-selective fading with a specified Power Delay Profile (PDP)
    As well as fast/slow time variance with a specified Doppler Spectrum
    '''
    def __init__(self, ip_signal, Fs=1, Fc = None, fading_process='Rayleigh', 
                 rho=None, F_rho=None, theta_rho=None,
                 doppler_spectrum='Jakes', max_doppler_shift=100, num_sinusoids=200,
                 path_delays=None, avg_path_gains=None, norm_path_gains=False):
        
        self.ip_signal = ip_signal #input signal
        self.num_samples = len(self.ip_signal)
        self.Fs = Fs #sampling frequency
        self.Ts = 1/Fs #sampling time
        self.Fc = Fc #carrier frequency

        #Channel Envelope Distribution
        self.fading_process = fading_process.upper()

        #Rician distribution specific parameters
        self.rho = rho #amplitude of LoS component 
        self.F_rho = F_rho #Doppler frequency of LoS component 
        self.theta_rho = theta_rho #phase of LoS component

        #time-variance specific parameters - specifying Doppler Spectrum
        self.doppler_spectrum = doppler_spectrum.upper() #doppler spectrum shape for all channel paths
        self.Fd_max = max_doppler_shift
        self.Tc = 1/(4*self.Fd_max) #coherence time
        self.num_samples_Tc = round(self.Tc * self.Fs) #number of samples corresponding to coherence time
        self.num_CIR = self.num_samples // self.num_samples_Tc #number of time channel changes in observation time
        self.num_sinusoids = num_sinusoids #Number of sinusoids used to model the fading process

        #Frequency-selective specific parameters - specifying Power Delay Profile (PDP)
        #path delays and gains can be time varying (2D array) or block faded (1D array)
        #NxM: N number of instances, M number of taps
        if (path_delays is not None):
            if (not(path_delays.shape == avg_path_gains.shape)):
                raise ValueError('Path Delays & Path Gains must have same dimensions')

            #Discrete path delay in seconds
            self.path_delays = np.tile(path_delays, (self.num_CIR,1)) if path_delays.shape[0] == 1 else path_delays 
            #Average gains of the discrete paths in decibels
            self.avg_path_gains_dBm = np.tile(avg_path_gains, (self.num_CIR,1)) if avg_path_gains.shape[0] == 1 else avg_path_gains 
            self.norm_path_gains = norm_path_gains #Normalize average path gains
        else: #None
            self.path_delays = path_delays
            self.avg_path_gains_dBm = avg_path_gains
            self.norm_path_gains = norm_path_gains

        #Computed Channel waveforms
        self.t = self.Ts * np.arange(0, self.num_samples) #time vector
        self.CIR = None
        self.Ch_env = None #complex channel coefficients

    def generate_TapGain_process(self, time_vec):
        '''
        Generates the tap-gain processes samples according to specified fading process & doppler spectrum
        The method of generation is the Sum-of-Sinusoids method (SOS)
        Model parameters are computed using  the method of exact Doppler spread (MEDS)
        '''
        def param_MEDS(self):
            '''
            Generates model parameters using the method of exact Doppler spread (MEDS)
            namely, : ci,n: Doppler coefficients, fi: discrete Doppler frequencies and θi: Doppler phases
            '''
            n = np.arange(0, self.num_sinusoids)

            c_i_n = np.sqrt(2/self.num_sinusoids)
            theta_i_n= 2 * np.pi * np.random.rand(self.num_sinusoids)

            if (self.doppler_spectrum == "JAKES"):
                f_i_n = self.Fd_max * np.sin(np.pi/(2*self.num_sinusoids)*(n-1/2))

            elif (self.doppler_spectrum == "GAUSSIAN"):
                f_i_n = self.Fc / np.sqrt(np.log(2)) * erfinv((2*n-1)/(2*self.num_sinusoids))

            return c_i_n, f_i_n, theta_i_n

        #generate model parameters
        [c_1,f_1,theta_1] = param_MEDS(self)
        [c_2,f_2,theta_2] = param_MEDS(self)
        #compute processes for in-phase & quadrature components
        z_i = 0
        z_q = 0
        for i in range(self.num_sinusoids):
            z_i = z_i + (c_1 * np.cos((2 * np.pi * time_vec * f_1[i]) + theta_1[i]))
            z_q = z_q + (c_2 * np.cos((2 * np.pi * time_vec * f_2[i]) + theta_2[i]))
        
        #add Line-of-Sight LoS component for Rician process
        if (self.fading_process == "RICIAN"):
            z_i = z_i + (self.rho * np.cos(2*np.pi*self.F_rho * time_vec + self.theta_rho))
            z_q = z_q + (self.rho * np.sin(2*np.pi*self.F_rho * time_vec + self.theta_rho))
        
        z = z_i + (1j * z_q)
        
        return z #z is a vector of complex samples, correlated in time and with the specified doppler PSD
    
    def TDL(self): 
        '''
        Generates the Channel Impulse Response (CIR) vector using a Tapped-Delay Line Model for the channel
        The TDL representation has a sampling rate equal to that of the input signal
        '''
        avg_path_gains_lin = 10 ** (self.avg_path_gains_dBm / 10) 
        max_delay_sample = round(np.max(self.path_delays) * self.Fs)

        #setting up channel envelope
        time_vec = np.zeros(self.num_samples)
        for n in range(self.num_CIR):
            stride = self.num_samples_Tc * n
            time_vec[stride:stride+self.num_samples_Tc] = stride

        unique_idx = np.unique(time_vec)

        self.Ch_env = np.zeros((self.num_samples, self.path_delays.shape[1]), dtype=complex)
        temp = self.Ch_env = np.zeros((self.num_samples, self.path_delays.shape[1]), dtype=complex)
        for i in range(len(self.path_delays[0,:])):
            self.Ch_env[:,i] = self.generate_TapGain_process(time_vec)
            temp[:,i] = self.Ch_env[:,i] * avg_path_gains_lin[0,i]

        #setting up CIR
        H = np.zeros((self.num_CIR, max_delay_sample + 1))
        for n in range(self.num_CIR):
            delay_samples = np.round(self.path_delays[n,:] * self.Fs).astype(int).reshape(-1)
            H[n, delay_samples] = temp[int(unique_idx[n]), :]

        return H

    def filter_CIR(self, CIR):
        '''
        Takes the Channel Impulse Response and convolves it (same convolution) 
        with the input signal to produce the output signal
        The CIR is a 2D array with every row representing the channel profile at a time instant
        If the CIR is 1D array the channel is said to be in Block fading
        '''

        len_kernel = len(CIR[0,:])

        padding = (len_kernel - 1) // 2 # Calculate the number of zeros to pad on each side of the signal
        padded_signal = np.pad(self.ip_signal, (len_kernel - 1 - padding, padding), 'constant') # Pad the signal with zeros

        output = np.zeros_like(self.ip_signal)

        # Perform convolution
        for i in range(self.num_samples):
            kernel_index = i // self.num_samples_Tc #changing kernel every coherence time
            output[i] = np.dot(padded_signal[i:i+len_kernel], CIR[kernel_index,:][::-1])

        return output
    
    def visualize(self, plot=None):
        '''
        This function helps with visualizing key waveforms of the wireless channel
        Channel Impulse Response (CIR)
        Channel Frequenxy Response
        Channel Envelope
        Doppler Spectrum
        Channel Envelope Histogram
        '''
        func_dict = {"IMPULSE RESPONSE": plot_CIR,
                     "CHANNEL ENVELOPE": plot_ch_env,
                     "CHANNEL FREQUENCY RESPONSE": plot_freq_res,
                     "DOPPLER SPECTRUM": plot_doppler_spectrum,
                     "CHANNEL ENVELOPE HISTOGRAM": plot_env_hist}
        
        func_dict[plot.upper()](self)
    
    def run(self):

        if (self.path_delays is None): #flat-fading channel
            self.Ch_env = self.generate_TapGain_process(self.t).reshape(-1,1) #complex channel coefficients
            op_signal = np.abs(self.Ch_env) * self.ip_signal
            self.CIR = np.abs(self.Ch_env).reshape(-1,1)

            return op_signal
        
        elif (self.path_delays.size > 1): #frequency-selective channel
            self.CIR = self.TDL()
            op_signal = self.filter_CIR(self.CIR)

            return op_signal
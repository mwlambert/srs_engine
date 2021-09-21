# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 10:58:47 2021

@author: m.lambert
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal

G_MPS = 9.807

plt.style.use('classic')

def natural_freq_array(start=10, end=10000, oct_steps=12):
    print('Generating natural frequency array with...\nStart Frequency: {} Hz\nEnd Frequency: {} kHz\nSteps Per Octave: {}\n'.format(start,end,oct_steps) )
    fn_array = [start]
    for i in range(end-start):
        f0 = start * 2**(1/oct_steps)
        fn_array.append(f0)
        start = f0
        if start > end:
            break
    return np.array(fn_array)
 
class SRS:
    def __init__(self, data):
        print('Bulding SRS object for given data. Run one of the SRS.calc_ methods to generate a response spectrum for the given input\n')
        # self.data = data
                     
        self.time = data.index.values
        
        # self.input_channel = data.iloc[:].name
        self.input_accel = data.iloc[:,0].values
        self.input_vel = integrate.cumtrapz(self.input_accel*G_MPS, self.time, initial=0.)
        self.input_disp = integrate.cumtrapz(self.input_vel, self.time, initial=0.)
        
        self.PLOT_MAXIMAX = False    
    
    def calc_kellyrichman_srs(self, fn_array, Q = 10, model='absolute'):
        """
        Computes the shock response spectrum for a given base excitation using the Kelly & Richman method.
        
        Parameters
        ----------
        fn_array : float
            Array of natural frequencies for assessing shock response. Defaults to 1 to 10k with 1/6 octave steps.
        Q : int, optional
            Q factor for determining damping of system. The default is 10.
        model : string, optional
            Defines which acceleration model to use. The absolute model is most frequently used for SRS cal
            culations. The relative displacement model is useful when the potential damage is related to
            the relative displacement between input and out. The default is 'absolute'.

        Returns
        -------
        None.

        """
        print('Calculing response spectrum...\n')
        self.method = 'Kelly & Richman'
        self.Q = Q
        self.fn_array = fn_array
        self.T = np.diff(self.time).mean()
        self.fs = 1/self.T
            
        self.zeta = 1/(2*self.Q)
        self.omega_n = 2 * np.pi * self.fn_array
        self.omega_d = self.omega_n * np.sqrt(1-self.zeta**2)
        
        cos_omegad_dt = np.cos(self.omega_d * self.T)
        sin_omegad_dt = np.sin(self.omega_d * self.T)
        z_omegan_dt = self.zeta*self.omega_n*self.T
                
        a0 = np.ones_like(self.fn_array)
        a1 = -2*np.exp(-z_omegan_dt) * cos_omegad_dt
        a2 = np.exp(-2*z_omegan_dt)
      
        b0 = 2*z_omegan_dt
        b1 = self.omega_n * self.T * np.exp(-z_omegan_dt)
        b1 = b1 * ( (self.omega_n/self.omega_d)*(1-2*(self.zeta**2)) * sin_omegad_dt - 2*self.zeta*cos_omegad_dt)
        b2 = np.zeros_like(self.fn_array)
        
        self.b = np.array([b0,b1,b2]).T
        self.a = np.array([a0,a1,a2]).T
                
        self.output_accels = [0]*len(fn_array)
        self.pos_accel = np.zeros_like(self.fn_array)
        self.neg_accel = np.zeros_like(self.fn_array)
        self.std_accel = np.zeros_like(self.fn_array)
        for i, f_n in enumerate(self.fn_array):
            out_accel = signal.lfilter(self.b[i], self.a[i], self.input_accel, axis=-1, zi=None) 
            self.output_accels[i] = out_accel
            self.pos_accel[i] = out_accel.max()
            self.neg_accel[i] = np.abs(out_accel.min())
            self.std_accel[i] = out_accel.std()
        
        # corrected=False
        # if corrected:
            # cos_omegad_dt = np.cos(self.omega_d * self.T)
            # sin_omegad_dt = np.sin(self.omega_d * self.T)
            # z_omegan_dt = self.zeta*self.omega_n*self.T
            # z_by_sqrt_1mz2 = self.z/(np.sqrt(1-self.z**2))
                        
            # S_k = self.input_accel[i+1] = self.input_accel[i]
            # S2_km1 = self.input_accel[i+1] = 2*self.input_accel[i] + self.input_accel[i-1]
            
            # B1 = np.exp(-z_omegan_dt) * (cos_omegad_dt + z_by_sqrt_1mz2*sin_omegad_dt)
            # B2 = (np.exp(-z_omegan_dt)/self.omega_d) * sin_omegad_dt
            # B3 = (-1/self.omega_n**2) - (1-*B1)
            # B4 = -1/self.omega_n**2 * (  (1-np.exp(-z_omegan_dt)) - ( ((1-(2*self.zeta**2)) * np.exp(-z_omegan_dt)*sin_omegad_dt)/(self.omega_d*self.dt) )  )
            
            
            
            # B5 = (-1/2*self.omega_n**2) * ( (I3/self.T**2) - (I2/self.T) )
            # B6 = -self.omega_n * B2
            # B7 = 
            # B8 = -B2/self.omega_n
            # B9 = (B1-1)/((self.omega_n**3)*self.T)
            # B10 = 
    
    def calc_smallwood_srs(self, fn_array, Q = 10, model = 'absolute'):
        """
        Computes the shock response spectrum for a given base excitation using the Smallwood ramp invariant
        digital recursive method from
        
        "AN IMPROVED RECURSIVE FORMULA FOR CALCULATING SHOCK RESPONSE SPECTRA"
        http://www.vibrationdata.com/ramp_invariant/DS_SRS1.pdf
        
        Parameters
        ----------
        fn_array : float
            Array of natural frequencies for assessing shock response. Defaults to 1 to 10k with 1/6 octave steps.
        Q : int, optional
            Q factor for determining damping of system. The default is 10.
        model : string, optional
            Defines which acceleration model to use. The absolute model is most frequently used for SRS cal
            culations. The relative displacement model is useful when the potential damage is related to
            the relative displacement between input and out. The default is 'absolute'.

        Returns
        -------
        None.

        """
        print('Calculing response spectrum...\n')
        self.method = 'Smallwood'
        self.model = model
        self.Q = Q
        self.fn_array = fn_array
        self.T = np.diff(self.time).mean()
        self.fs = 1/self.T
            
        self.zeta = 1/(2*self.Q)
        self.omega_n = 2 * np.pi * self.fn_array
        self.omega_d = self.omega_n * np.sqrt(1-self.zeta**2)
        E = np.exp(-self.zeta*self.omega_n*self.T)
        K = self.omega_d*self.T
        C = E*np.cos(K)
        S = E*np.sin(K)
        S_prime = S/K
        
        b0 = 1 - S_prime
        b1 = 2 * (S_prime-C)
        b2 = E**2 - S_prime
        
        if self.model == 'relative':
            A = 1/(self.T*self.omega_n)
            B = ((2*self.zeta**2 -1)*S) / np.sqrt(1-self.zeta**2)
            
            b0 = A * (self.zeta*(C-1) + B + self.T*self.omega_n)
            b1 = A * ((-2*C*self.T*self.omega_n) + 2*self.zeta*(1-E**2) - 2*B)
            b2 = A * (E**2 * (self.T*self.omega_n + 2*self.zeta) - 2*self.zeta*C + B)
        
        a0 = np.ones_like(self.fn_array)
        a1 = -2*C
        a2 = E**2
        
        self.b = np.array([b0,b1,b2]).T
        self.a = np.array([a0,a1,a2]).T
        
        self.output_accels = [0]*len(fn_array)
        self.pos_accel = np.zeros_like(self.fn_array)
        self.neg_accel = np.zeros_like(self.fn_array)
        self.std_accel = np.zeros_like(self.fn_array)
        for i, f_n in enumerate(self.fn_array):
            out_accel = signal.lfilter(self.b[i], self.a[i], self.input_accel)
            self.output_accels[i] = out_accel
            self.pos_accel[i] = out_accel.max()
            self.neg_accel[i] = np.abs(out_accel.min())
            self.std_accel[i] = out_accel.std()
        
        # self.plot_srs()
            
    def plot_srs(self, ax=0):
        if not ax:
            fig, ax = plt.subplots(figsize=(18,9))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Acceleration (g)')
        ax.set_title('{} Method Q = {}'.format(self.method,self.Q))
        ax.set_xlim(self.fn_array.min(),self.fn_array.max())
        ax.loglog(self.fn_array, self.pos_accel, label='Response +ve')
        ax.loglog(self.fn_array, self.neg_accel, label='Response -ve')
        if self.PLOT_MAXIMAX:
            ax.loglog(self.fn_array, self._calc_maximax(), label='Maximax', color = 'r', linestyle='--')
        ax.grid(True, which='both')
        ax.legend(loc='right', prop={'size': 12})
        
    def _plot_input_accel(self, ax=0):
        if not ax:
            fig, ax = plt.subplots(figsize=(18,9))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.set_title('Base Input: Acceleration')
        ax.plot(self.time, self.input_accel, label='Input Pulse', color='k', linewidth=1)
        ax.grid(True, which='both')
        
    def _plot_input_velocity(self, ax=0):
        if not ax:
            fig, ax = plt.subplots(figsize=(18,9))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Base Input: Velocity')
        ax.plot(self.time, self.input_vel, label='Input Pulse', color='fuchsia')
        ax.grid(True, which='both')
        
    def _plot_input_displacement(self, ax=0):
        if not ax:
            fig, ax = plt.subplots(figsize=(18,9))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (m)')
        ax.set_title('Base Input: Displacement')
        ax.plot(self.time, self.input_disp, label='Input Pulse')
        ax.grid(True, which='both')
    
    def _plot_output_accel(self, ax=0):
        if not ax:
            fig, ax = plt.subplots(figsize=(18,9))
            ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.set_title('Acceleration Response Q = {}'.format(self.Q))
        ax.plot(self.time, self.input_accel, label='Input Pulse', color='k', linewidth=1)
        for i, f_n in enumerate(self.fn_array):
            ax.plot(self.time, self.output_accels[i], label='Response of {:.1f} Hz System'.format(f_n))
        ax.grid(True, which='both')
        ax.legend(loc='lower right', prop={'size': 10})
    
    def _plot_output_velocity(self, ax=0):
        if not ax:
            fig, ax = plt.subplots(figsize=(18,9))
            ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Response Q = {}'.format(self.Q))
        self.output_vels = []
        for i, f_n in enumerate(self.fn_array):
            self.output_vels.append(integrate.cumtrapz(self.output_accels[i]*G_MPS, self.time, initial=0.))
            ax.plot(self.time, self.output_vels[i], label='Response of {:.1f} Hz System'.format(f_n))
        ax.grid(True, which='both')
        ax.legend(loc='lower right', prop={'size': 10})
    
    def plot_small_data(self):
        fig, axes = plt.subplots(2, 2, figsize=(18,9))
        fig.suptitle('{} Method'.format(self.method), fontsize=14)
                
        axes[0,0] = self._plot_output_accel(axes[0,0])
        axes[1,0] = self._plot_input_accel(axes[1,0])
        axes[0,1] = self._plot_output_velocity(axes[0,1])
        axes[1,1] = self._plot_input_velocity(axes[1,1])
        plt.show()
    
        # self._plot_input_displacement()
        
    def plot_results(self):
        fig = plt.figure(figsize=(18,9))
        ax1 = plt.subplot(2,2,(1,2))
        ax2 = plt.subplot(2,2,3)
        ax3 = plt.subplot(2,2,4)
        
        self.PLOT_MAXIMAX = True
        ax1 = self.plot_srs(ax1)
        ax2 = self._plot_input_accel(ax2)
        ax3 = self._plot_input_velocity(ax3)
        plt.show()
                        
        
    def _calc_maximax(self):
        maximax = []
        for accel in self.output_accels:
             maximax.append(np.abs(accel).max())
             
        return np.array(maximax)


def create_half_sine(A=1000, T=0.01, num_half_sins=1):
    length = 0.04
    x = np.arange(0, length, 1e-5)
    fx = A * np.sin(np.pi*x / T)
    fx[x > num_half_sins*T] = 0
    return x, fx

def create_sawtooth(A=1000, T=0.01):
    length = 0.04
    x = np.arange(0, length, 1e-5)    
    fx = A * x
    fx[x > T] = 0
    return x, fx

def create_step_input(A=1000, T=0.01):
    length = 0.04
    x = np.arange(0, length, 1e-5)    
    
    fx = np.zeros_like(x)
    fx[10:10+int(T/1e-5)]=1*A
    
    return x, fx

if __name__ == '__main__':
    import pandas as pd    
    fn_array = np.array([10,100,200])
    x, y = create_half_sine(A=1000, T=0.008, num_half_sins=4)
    df = pd.DataFrame({'time':x, 'data':y})
    df=df.set_index('time')
    srs=SRS(df)
   
    # Kelly & Richman
    srs.calc_kellyrichman_srs(fn_array=natural_freq_array())
    srs.plot_results()
    srs.calc_kellyrichman_srs(fn_array=fn_array)
    srs.plot_small_data()
    
    # Smallwood
    # srs.calc_smallwood_srs(fn_array=natural_freq_array())
    # srs.plot_results()
    # srs.calc_smallwood_srs(fn_array=fn_array)
    # srs.plot_small_data()

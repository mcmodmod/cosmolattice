from scipy.integrate import quad
import numpy as np
from scipy.integrate import trapezoid, simpson


class GW_analysis:
    def __init__(self,m_over_H,H_star,T_rh,g_star=106.75):
        '''
        load axion dark photon constraint class
        '''
        #axionclass = AxionConstraints(f,alpha,theta)

        self.m_over_H = m_over_H
        self.m = self.m_over_H * H_star
        self.H_star = H_star
        self.T0 = 2.352531980526e-13 #GeV
        self.T_rh = T_rh
        self.g_star = g_star
        self.g_0 = 3.91
        self.redshift_freq_factor = 1.65e-7 * self.T_rh * (self.g_star/100)**(1/6) 
        self.redshift_amplitude_factor = 1.67e-5 * (100/self.g_star)**(1/3) 

    def GW_template(self, f):
        '''
        PUT IN GW TEMPLATE
        ''' 
        return 


    def SNR(self,t_obs,sens_curve,fmin,fmax):
        '''
        computes SNR (sensitivity curve of experiment is an input)
        '''
        #res = quad(lambda f: (self.GW_template(f)/sens_curve(f))**2, fmin, fmax, epsrel = 1e-20,epsabs=1e-20)[0]
        x = np.logspace(np.log10(fmin),np.log10(fmax),10000)
        sens = np.array([sens_curve(i) for i in x])
        y = (self.GW_template(x)/sens)**2
        res = simpson(y,x)
        #simp_result = simpson(y, x)
        return np.sqrt(2 * t_obs * res)

    def SNR_auto(self,t_obs,sens_curve,fmin,fmax):
        '''
        computes SNR (auto-correlated)
        '''
        #res = quad(lambda f: (self.GW_template(f)/sens_curve(f))**2, fmin, fmax, epsrel = 1e-30,epsabs=1e-30)[0]
        x = np.logspace(np.log10(fmin), np.log10(fmax), 10000)
        sens = np.array([sens_curve(i) for i in x])

        y = (self.GW_template(x) / sens) ** 2
        res = simpson(y, x)
        return np.sqrt(t_obs * res)
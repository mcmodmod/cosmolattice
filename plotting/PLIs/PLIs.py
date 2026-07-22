from GW_detectors import *

class PLI_curves:

    def __init__(self, fmin, fmax, minlogfreq, maxlogfreq, pmin, pmax, name, t_obs, rho_thresh, N_freq):
        import numpy as np
        

        self.PLIs = []
        self.omega_threshs = []
        self.fmin = fmin #lower integration boundary
        self.fmax = fmax #upper integration boundary
        #self.powers = np.arange(pmin,pmax,1) #power parameter space
        self.powers = np.linspace(pmin,pmax,100) #power parameter space

        self.name = name #which detector
        self.t_obs = t_obs #observation time
        self.rho_thresh = rho_thresh #detection threshold
        
        self.et_data = np.genfromtxt('ET-0000A-18_ETDSensitivityCurveTxtFile.txt')
        self.freq = np.array([a[0] for a in self.et_data])
        
        #if self.name == 'et':
        #    self.fs = self.freq
        #else:
        self.fs = np.logspace(minlogfreq, maxlogfreq, N_freq)
            
        self.GW_detector = GW_detectors(name)

            
    def power_law(self, freq, p):
        if self.name == 'ska':
            f0 = (365 * 24 * 3600)**(-1)
            return (freq/f0)**p
        elif self.name == 'decigo':
            f0 = 7.36
            return (freq/f0)**p  
        elif self.name == 'bbo':
            f0 = 7.36
            return (freq/f0)**p  
        else:
            f0 = 1e-3
            return (freq/f0)**p
            
    
    def spec_ratio(self, freq, p):
        #from GW_detectors import GW_detectors
        #return (self.power_law(freq, p)/GW_detectors(self.name, freq).spectral_density())**2
        return (self.power_law(freq, p)/self.GW_detector.spectral_density(freq))**2

    
    
    def calculate_omega_threshs(self):
        import scipy.integrate as integrate
        import numpy as np
        
        if self.name != 'et':
            for p in self.powers:
                SNR_thresh = (integrate.quad(self.spec_ratio, self.fmin, self.fmax, args = (p),epsabs=1e-15,epsrel=1e-25)[0])**(-1/2)
                self.omega_threshs.append(SNR_thresh)  
        else:
            for p in self.powers:
                SNR_thresh = 0
                for i in range(1,len(self.fs)):
                    delta_f = self.fs[i] - self.fs[i-1]
                    SNR_thresh += delta_f * self.spec_ratio(self.fs, p)[i]
                self.omega_threshs.append(SNR_thresh**(-1/2))
        
        return self.omega_threshs

    def maximise_p(self):
        import numpy as np
        omegas = self.calculate_omega_threshs()
        #print(self.fs)
        print('omega computation done')
        
        if self.name == 'lisa':
            for freq in self.fs:
                temp = []
                for p in range(len(self.powers)):
                    temp.append((self.rho_thresh /np.sqrt(self.t_obs * 365 * 24 * 3600)) * omegas[p] * self.power_law(freq, self.powers[p]))                   
                self.PLIs.append(max(temp))
            fs = [self.fmin - 1e-20] + list(self.fs)
            PLIs = [1e2] + self.PLIs

            return fs, PLIs
        
        elif self.name =='et':
            for freq in self.fs[1:]:
                temp = []
                for p in range(len(self.powers)):
                    temp.append((self.rho_thresh /np.sqrt(self.t_obs * 365 * 24 * 3600)) * omegas[p] * self.power_law(freq, self.powers[p]))                   
                self.PLIs.append(max(temp))
            fs = [self.fmin - 1e-20] + list(self.fs[1:])
            PLIs = [1e2] + self.PLIs

            return fs, PLIs
        
        elif self.name == 'ska':
            for freq in self.fs:
                temp = []
                for p in range(len(self.powers)):
                    temp.append((self.rho_thresh /np.sqrt(2 * self.t_obs * 365 * 24 * 3600)) * omegas[p] * self.power_law(freq, self.powers[p]))
                self.PLIs.append(max(temp))
                
            fs = [f for f in self.fs if f > self.fmin]
            PLIs = [self.PLIs[i] for i in range(len(self.PLIs)) if self.fs[i] > self.fmin]
            fs = [self.fmin - 1e-20] + list(fs) + [self.fmax + 1e-20]
            PLI_Final = [1e2] + PLIs + [1e2]
            return fs, PLI_Final

        
        else:
            for freq in self.fs:
                temp = []
                for p in range(len(self.powers)):
                    temp.append((self.rho_thresh /np.sqrt(2 * self.t_obs * 365 * 24 * 3600)) * omegas[p] * self.power_law(freq, self.powers[p]))
                self.PLIs.append(max(temp))
            fs = [self.fmin - 1e-20] + list(self.fs) + [self.fmax + 1e-20]
            PLIs = [1e2] + self.PLIs + [1e2]
            return fs, PLIs

'''
Author: Mario Gennaro

Try to create a TSMC class.
Let's see how this goes
'''

import numpy as np
import emcee, corner, sys
from scipy.optimize import brentq
from matplotlib import pyplot as plt
import pandas as pd

class TsmcSampler:
    
    def __init__(self,TSMCpars):
  
        #assign the variables values
        self.starnum  = TSMCpars['starnum']
        self.AIsteps  = TSMCpars['AIsteps']
        self.AIfact   = TSMCpars['AIfact']
        self.limits   = TSMCpars['limits']
        self.tau      = TSMCpars['tau']
        self.loglik   = TSMCpars['loglik']         
        self.logqbt   = TSMCpars['logqbt'] 
        self.logpri   = TSMCpars['logpri']
        self.logpst   = TSMCpars['logpst']
        self.logptm   = TSMCpars['logptm']
        self.argsqbt  = TSMCpars['argsqbt']
        self.argspri  = TSMCpars['argspri']
        self.argslik  = TSMCpars['argslik']
        self.argspst  = TSMCpars['argspst']
        self.argsptm  = TSMCpars['argsptm']
        self.ndims    = TSMCpars['ndims']
        self.nwalkers = TSMCpars['nwalkers']
        
        #Some initial default parameters
        self.delta_bis = 0.
        self.delta_flr = 0.
        self.iteration = 0
        self.idx = np.arange(self.nwalkers)
        self.unique = 0.
        print('Calling the initializer for star:',self.starnum)
        sys.stdout.flush()
        self.P_initializer()	
        print('Done initializing for star:',self.starnum)
        sys.stdout.flush()
        
    def P_initializer(self):     
        #Get random uniform initial positions
        # and compute logq, loglik and logpr

        self.p0 = []
        for i in range(len(self.limits)):
            self.p0.append(np.random.uniform(self.limits[i][0],self.limits[i][1],self.nwalkers))   
        self.p0 = np.asarray(self.p0).T
       
        self.lq = np.empty(self.nwalkers)
        self.ll = np.empty(self.nwalkers)
        self.lp = np.empty(self.nwalkers)
        self.lw = np.empty(self.nwalkers)
        
        for i in range(self.nwalkers):
            self.lq[i]       = self.logqbt(self.p0[i],*self.argsqbt)
            self.lp[i]       = self.logpri(self.p0[i],*self.argspri)    
            self.ll[i],d1,d2 = self.loglik(self.p0[i],*self.argslik)


    def EF_delta_minus_tau(self,value):
        msk  = np.isfinite(self.lw)
        llw  = self.lw[msk]
        wgh  = np.exp((llw-np.amax(llw))*(value-self.delta_flr))
        wgh2 = wgh**2
        num  = (np.sum(wgh))**2
        den  = np.sum(wgh2)
        return 1./np.size(wgh) * num/den - self.tau


    def step1(self):
        #Step 1: compute new delta
        self.lw = self.ll+self.lp-self.lq
        EF_1mtau = self.EF_delta_minus_tau(1.)
        if (EF_1mtau < 0):
            self.delta_bis = brentq(self.EF_delta_minus_tau,self.delta_flr,1.)
        else:
            self.delta_bis = 1.

    def step2(self):
        #Step 2: resample with weights
        lwd = self.lw*(self.delta_bis-self.delta_flr)
        probs = np.exp(lwd-np.amax(lwd))
        probs = probs/np.sum(probs[np.where(~np.isnan(probs))])
        resam_idx = np.random.choice(self.idx, size=self.nwalkers, p=probs)
        self.unique = np.size(np.unique(resam_idx))
        self.p0 = self.p0[resam_idx]
        
    def step3(self):
        #Step 3: affine invariant move
        if(self.delta_bis == 1.):
            #Initialize the sampler to sample the true posterior (i.e. unsmoothed)
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpst,
                                                    args=self.argspst)
            pos, prob, state, blob = self.sampler.run_mcmc(self.p0, self.AIsteps * self.AIfact)

        else:
            #Initialize the sampler to sample the smooth posterior
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logptm,
                                                 args=[*self.argsptm,self.delta_bis])
            pos, prob, state, blob = self.sampler.run_mcmc(self.p0, self.AIsteps)
            self.p0 = pos

            for nw in range(self.nwalkers):
                self.lq[nw] = self.sampler.blobs[self.AIsteps-1][nw]['logq']
                self.ll[nw] = self.sampler.blobs[self.AIsteps-1][nw]['loglik']
                self.lp[nw] = self.sampler.blobs[self.AIsteps-1][nw]['logprior']
        
    def step4(self):
        #Step 4:
        self.delta_flr = self.delta_bis
    

    def run_TSMC(self):
        while (self.delta_flr < 1):
            print('Star: ',self.starnum,'Iteration:',self.iteration,'Starting step 1')
            sys.stdout.flush()
            self.step1()
            print('Star: ',self.starnum,'Iteration:',self.iteration,'Starting step 2')
            sys.stdout.flush()
            self.step2()
            print('Star: ',self.starnum,'Iteration:',self.iteration,'Starting step 3')
            sys.stdout.flush()
            self.step3()
            print('Star: ',self.starnum,'Iteration:',self.iteration,'Starting step 4')
            sys.stdout.flush()
            self.step4()
            print('Star: ',self.starnum,'Finished Iteration:',self.iteration,'Delta_bis:', self.delta_bis, 'Unique indices:',self.unique)
            sys.stdout.flush()
            self.iteration = self.iteration+1
        return self.sampler

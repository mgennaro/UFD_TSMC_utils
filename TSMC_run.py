'''
Author: Mario Gennaro
Run the TSMC algorithm on a series of stars from start to stop

Usage (from command line):

$ python TSMC_run star_start star_stop ncores

'''

import shelve, pickle, emcee, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.optimize import brentq
from scipy.stats import norm
from synCMD.auxfunc.photband import photband
from synCMD.generators.GeneralRandom import GeneralRandom
import extinction
from astropy import units as u
from scipy.io import readsav
from TSMC_utils.logpdf import loglik,logprior,logpost,logpost_tempered
from TSMC_utils.intNN import intNN
from TSMC_utils.TsmcSampler import TsmcSampler
from get_completeness import get_completeness
from joblib import Parallel, delayed
from itertools import combinations

########################
# This is the only part that need to be changed
# when running different galaxies or different use-cases
# i.e., the path to the setup file
########################

savepath='/user/gennaro/UFDs_OPT/herc/TSMCout/'

########################

'''
A wrap-up of all the things that need to be done for one star,
so that I can run multiple stars in parallel
'''
def onestar(pars,starnum,cat,AStree,dfAS,photbands,savepath):
    
    print('Starting star: ',starnum)
    sys.stdout.flush()
    ##########################
    # 1: run the TSMC part
    
    # 1a: add the current star parameters to the input dictionary
    pars['starnum'] = starnum
    star = cat.iloc[starnum]
    pars['argslik'][3] = star
    pars['argspst'][3] = star
    pars['argsptm'][3] = star
    
    # 1b: run the TSMC sampler
    sampler = TsmcSampler(pars)
    samples = sampler.run_TSMC()

    ##########################
    # 2: get the samples magnitudes, probability values and completeness

    mags_in = np.empty([pars['nwalkers'],len(photbands)])
    mags_ou = np.empty([pars['nwalkers'],len(photbands)])
    logpst  = np.empty([pars['nwalkers']])
    loglik  = np.empty([pars['nwalkers']])
    logpri  = np.empty([pars['nwalkers']])
    cmpval  = np.empty([pars['nwalkers']])

    for nw in range(pars['nwalkers']):
        for nb in range(len(photbands)):
            mags_in[nw,nb] = samples.blobs[-1][nw]['magsin'][nb]
            mags_ou[nw,nb] = samples.blobs[-1][nw]['magsou'][nb]
        logpst[nw]  = samples.lnprobability[nw][-1]
        loglik[nw]  = samples.blobs[-1][nw]['loglik']
        logpri[nw]  = samples.blobs[-1][nw]['logprior']
        cmpval[nw]  = get_completeness(np.array([mags_in[nw,:]]).reshape(1,-1),AStree,dfAS,photbands)

    ##########################
    # 4: save the results
    
    TSMCoutput = {'starnum':starnum,
                  'mags_in':mags_in,
                  'mags_ou':mags_ou,
                  'logpst':logpst,
                  'loglik':loglik,
                  'logpri':logpri,
                  'cmpval':cmpval,
                  'chain':samples.chain[:,-1,:]
                  }
    
    pickle.dump( TSMCoutput, open(savepath+'TSCMoutput_'+("%.0f" % starnum)+'.pickle', "wb" ) )  
    
    print('Done star: ',starnum)
    sys.stdout.flush()


#################
# Restore the setup file and reassign the variables
##################

TSMC_setup = pickle.load(open(savepath+'Setup.pickle', "rb" ))

photbands  = TSMC_setup['photbands']
limits_all = TSMC_setup['limits_all']
qGRlist_all= TSMC_setup['qGRlist_all']
GRlist_all = TSMC_setup['GRlist_all']
cols       = TSMC_setup['cols']
binsize    = TSMC_setup['binsize']
nwalkers   = TSMC_setup['nwalkers']
niters     = TSMC_setup['niters']
ndims      = TSMC_setup['ndims']
nfact      = TSMC_setup['nfact']
tau        = TSMC_setup['tau']
defaults   = TSMC_setup['defaults']
keys       = TSMC_setup['keys']
msk        = TSMC_setup['msk']
UFD        = TSMC_setup['UFD']
path       = TSMC_setup['path']


#################
#Restore the NN interpolator list
#################

iso_NNInt = pickle.load(open('/user/gennaro/UFDs_OPT/shelves/iso_NNint.pickle',"rb"))

#################
#Artificial stars part
#################

#read the AS catalog
f2r = path+UFD+'/'+UFD+'.art'
dfAS = pd.read_table(f2r,header=None,sep='\s+',
                   names=['F606W_in','F606W_out','F814W_in','F814W_out'])

dfAS['det_flag'] = msk[0,:] & msk[1,:]

#Restore the KD-Tree of the input AS magnitudes
magAS_IN  = np.asarray([dfAS[pb.name+'_in'] for pb in photbands]).T
magAS_OUT = np.asarray([dfAS[pb.name+'_out'] for pb in photbands]).T
AStree = pickle.load(open( '/user/gennaro/UFDs_OPT/shelves/AS_'+UFD+'.pickle', "rb" ) ) 

#################
# Restore the observed stars
#################

cat = pickle.load(open( '/user/gennaro/UFDs_OPT/shelves/cat_'+UFD+'.pickle', "rb" ) )   
cat.head()

#compute the likelihood normalization for convnience
lnorm = np.zeros(len(cat.index))
for pb in photbands:
    lnorm = lnorm - 0.5*np.log(2*np.pi) - np.log((cat['err_'+pb.name]))

cat['ln(norm)'] = lnorm

#################
# Restrict the priors only to the used dimensions
#################

GRlist = [GR for indx,GR in enumerate(GRlist_all) if keys[indx]]
qGRlist = [qGR for indx,qGR in enumerate(qGRlist_all) if keys[indx]]
limits = [lm for indx,lm in enumerate(limits_all) if keys[indx]]


#################
# The dictionary of parameters for the TSMC sampler
#################

TSMCpars = {'AIsteps':niters,
            'AIfact':nfact,
            'limits':limits,
            'tau':tau,
            'loglik':loglik,         
            'logqbt':logprior,
            'logpri':logprior,
            'logpst':logpost,
            'logptm':logpost_tempered,
            'argsqbt':[qGRlist],
            'argspri':[GRlist],
            'argslik':[keys,defaults,photbands,'dum',AStree,dfAS,iso_NNInt],
            'argspst':[keys,defaults,photbands,'dum',AStree,dfAS,iso_NNInt,GRlist],
            'argsptm':[keys,defaults,photbands,'dum',AStree,dfAS,iso_NNInt,GRlist,qGRlist],
            'ndims':ndims,
            'nwalkers':nwalkers
             }

#################
# The parallel run
#################

Parallel(n_jobs=int(sys.argv[3]))(delayed(onestar)(TSMCpars,starnum,cat,AStree,dfAS,photbands,savepath) 
                                    for starnum in range(int(sys.argv[1]),int(sys.argv[2])))





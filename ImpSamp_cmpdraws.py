'''
Author: Mario Gennaro

Purpose: uniformly draw model parameters in the allowed range
and compute the completeness.
Used for the normalization factor of the PPP 

'''

import shelve, pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KDTree
from synCMD.auxfunc.photband import photband
from synCMD.generators.GeneralRandom import GeneralRandom
from scipy.io import readsav
from get_completeness import get_completeness
from joblib import Parallel, delayed
from itertools import combinations
from TSMC_utils.logpdf import loglik

########################
# This is the only part that need to be changed
# when running different galaxies or different use-cases
# i.e., the path to the setup file
########################

savepath = '/user/gennaro/UFDs_OPT/herc/TSMCout/'
ndraws   = 500000   # Number of draws per core
ncores   = 10      # Number of parallel runs

########################
# Auxiliary function to compute the model magnitude
# given some paramters (mass,age,metallicity,DM,bf,extinction/colorexcess)
########################

def getmagmodel(passed,keys,defaults,photbands,iso_Int):
    
    # Assign to the array of parameters either the passed or the default values,
    # according to the keys array
    
    pars       = defaults
    pars[keys == True] = passed
    
    # Transform some of the paramters:
    # the mass parameter is passed as logarithmic value --> get the linear value
    # the extinction parameters are passed as color excesses and reddest A_lambda --> get all A_lambdas

    pars[0] = np.exp(pars[0])
    ext = np.zeros_like(pars[5:])
    for i,pp in enumerate(pars[5:]):
        ext[i] = np.sum(pars[5+i:])
    
    # Check whether this model is a binary and interpolate the isocrones to get the
    # intrinsic magnitude of the current model

    if (pars[4] >= 0):
        # If the system is binary get the mass ratio, obtain interpolated
        # magnitudes for each component and sum them
        mass_2  = pars[0]/(1.+pars[4])*np.array([1.,pars[4]],dtype=float)
        sysmags = np.empty([2,len(photbands)])
        for j,mcomp in enumerate(mass_2):
            for i in range(len(photbands)):
                sysmags[j,i] = iso_Int[i](mcomp,pars[1],pars[2])

        sysmags = 10**(-0.4*sysmags)
        modelmags = np.empty([len(photbands)])
        for j in range(2):
            oneband = sysmags[:,j]
            msk = ~np.isnan(oneband)
            if (~np.any(msk)):
                modelmags[j] = np.nan
            else:
                modelmags[j] = -2.5* np.log10(np.sum(oneband[msk]))
    else:
        modelmags = np.empty(len(photbands))
        for i in range(len(photbands)):
            modelmags[i] = iso_Int[i](pars[0],pars[1],pars[2])

    #If the model exists add DM and extinction

    if (~np.any(np.isnan(modelmags))):
        modelmags = modelmags + ext + pars[3]*np.ones_like(modelmags)

    return modelmags

#####################
# Make a chunck of draws and save them
#####################


def onechunk(ndims,ndraws,photbands,limits,keys,defaults,iso_NNInt,AStree,dfAS,j):

# Draw samples and normalize them
# in the right intrervals

    samples = np.random.uniform(size=(ndims,ndraws))

    for i in range(ndims):
        samples[i,:] = limits[i][0] + (limits[i][1] - limits[i][0]) * samples[i,:]

        comp = np.zeros(ndraws)
        mags = np.zeros([ndraws,len(photbands)])

    for i in range(ndraws):
        if (i%100 == 0):
            print('Doing draw:',i,' for chunk ',j)
            sys.stdout.flush()
        modelmags = getmagmodel(samples[:,i],keys,defaults,photbands,iso_NNInt)

        if (np.any(np.isnan(modelmags))):
            comph = 0.
        else:
            comph = get_completeness(modelmags.reshape(1,-1),AStree,dfAS,photbands)

            mags[i,:] = modelmags 
            comp[i]  = comph

    dictosave = {'samples':samples,
                 'mags':mags,
                 'comp':comp}

    pickle.dump( dictosave, open(savepath+'Compdraws_'+str(j)+'.pickle', "wb" ) )  



########################
# Restore the setup file and reassign the variables
########################

TSMC_setup = pickle.load(open(savepath+'Setup.pickle', "rb" ))

photbands  = TSMC_setup['photbands']
limits_all = TSMC_setup['limits_all']
cols       = TSMC_setup['cols']
binsize    = TSMC_setup['binsize']
defaults   = TSMC_setup['defaults']
ndims      = TSMC_setup['ndims']
keys       = TSMC_setup['keys']
msk        = TSMC_setup['msk']
UFD        = TSMC_setup['UFD']
path       = TSMC_setup['path']

limits = [lm for indx,lm in enumerate(limits_all) if keys[indx]]

#######################
#Artificial stars part
#######################

#read the AS catalog
f2r = path+UFD+'/'+UFD+'.art'
dfAS = pd.read_table(f2r,header=None,sep='\s+',
                   names=['F606W_in','F606W_out','F814W_in','F814W_out'])

dfAS['det_flag'] = msk[0,:] & msk[1,:]

#Restore the KD-Tree of the input AS magnitudes
magAS_IN  = np.asarray([dfAS[pb.name+'_in'] for pb in photbands]).T
magAS_OUT = np.asarray([dfAS[pb.name+'_out'] for pb in photbands]).T
AStree = pickle.load(open( '/user/gennaro/UFDs_OPT/shelves/AS_'+UFD+'.pickle', "rb" ) ) 

#######################
#Restore the NN interpolator list
#######################

iso_NNInt = pickle.load(open('/user/gennaro/UFDs_OPT/shelves/iso_NNint.pickle',"rb"))

    
#######################
# The parallel run
#######################


Parallel(n_jobs=ncores)(delayed(onechunk)(ndims,ndraws,photbands,limits,keys,defaults,iso_NNInt,AStree,dfAS,j) 
                                    for j in range(ncores))

samples = []
mags = []
comp = []

for j in range(ncores):
    dictoload = pickle.load( open(savepath+'Compdraws_'+str(j)+'.pickle', "rb" ) )
    samples.append(dictoload['samples']) 
    mags.append(dictoload['mags'])
    comp.append(dictoload['comp'])
    os.remove(savepath+'Compdraws_'+str(j)+'.pickle')

samples = (np.hstack(samples)).T
mags = np.vstack(mags)
comp = np.hstack(comp)

dictosave = {'samples':samples,
             'mags':mags,
             'comp':comp}


pickle.dump( dictosave, open(savepath+'Compdraws.pickle', "wb" ) )  

    

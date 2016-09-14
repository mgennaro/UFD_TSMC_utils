'''
Author: Mario Gennaro
This module contains functions to compute
log likelihood, log prior and log posterior
for comparison of stellar models to stellar photometry

These functions are intended to work with the emcee code
'''

'''
This function computes the loglikelihood for a given set of parameters
which include [Mass, Age, [Fe/H], DM, BinaryMassFraction,ExtinctionValue(one per band)].

As auxiliary inputs, It needs a list of photbands objects,
a pandas datafrmae entry for the selected star,
the scikit-learn KDETree of artificial stars,
the pandas dataframe of artificial stars,
the isochronen scipy Linear ND interpolator object 
(which interpolates in mass,age,metallicity)

'''

import numpy as np
from synCMD.auxfunc.photband import photband

def loglik(passed,keys,defaults,photbands,star,AStree,dfAS,iso_Int):

    # Assign to the array of parameters either the passed or the default values,
    # according to the keys array
    
    pars       = defaults
    pars[keys == True] = passed
    
    #Transform some of the paramters:
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
    
    # Find the corresponding output magnitude
    if (~np.any(np.isnan(modelmags))):
        modelmags = modelmags + ext + pars[3]*np.ones_like(modelmags)
        
        #then find the closest artificial star
        d,ind = AStree.query(modelmags.reshape(1,-1))
    
        #get the output magnitudes and chisq
        outmags =  np.empty(len(photbands))
        if (dfAS.iloc[ind[0]]['det_flag'].values == True):
            chisq = 0.
            for i,pb in enumerate(photbands):
                outmags[i] = dfAS.iloc[ind[0]][pb.name+'_out'].values + modelmags[i] - dfAS.iloc[ind[0]][pb.name+'_in'].values
                chisq  = chisq + ((star[pb.name]-outmags[i])/star['err_'+pb.name])**2
        else:
            chisq = np.inf
            outmags[:] = np.nan
    else:
        chisq = np.inf
        outmags = np.nan*np.ones(len(photbands))
        
    return -0.5*chisq+star['ln(norm)'],modelmags,outmags


    '''
Author: Mario Gennaro

This function computes the logprior probability
for a set of parameters.

It needs a np.ndarray of parameters and a list
(of the same lenght) of GenericRandom objects
'''

def logprior(passed,GRlist):
    lnp = 1.
    for i,GR in enumerate(GRlist):
        lnp = lnp*GR.getpdf(passed[i])

    if (lnp == 0.):
        return -1*np.inf
    else:
        return np.log(lnp)


'''
This function computes calls the log likelihood
and the log prior functions and returns the log posterior
it also rearranges a set of output paramters that can be
useful in postprocessing. When used with emcee those parameters
are returned in the blob object.

It needs both sets of parameters needed by the log likelihood
and log prior functions
'''


def logpost(passed,keys,defaults,photbands,star,AStree,dfAS,iso_Int,GRlist):
    lp = logprior(passed,GRlist)
    ll,modelmags,outmags = loglik(passed,keys,defaults,photbands,star,AStree,dfAS,iso_Int)
            
    auxout = {'magsin':modelmags,
              'magsou':outmags,
              'loglik':ll,
              'logprior':lp}

    return ll+lp, auxout
    

'''
This function calls the log likelihood, the log prior and the logQ (TSMC)
functions and returns the tempered log posterior
it also rearranges a set of output paramters that can be
useful in postprocessing. When used with emcee those parameters
are returned in the blob object.

It needs the 3 sets of parameters needed by the log likelihood
and log prior functions as well as th delta for their linear combination
'''


def logpost_tempered(passed,keys,defaults,photbands,star,AStree,dfAS,iso_Int,GRlist,qGRlist,delta):
    lp = logprior(passed,GRlist)
    lq = logprior(passed,qGRlist)
    ll,modelmags,outmags = loglik(passed,keys,defaults,photbands,star,AStree,dfAS,iso_Int)
            
    auxout = {'magsin':modelmags,
              'magsou':outmags,
              'loglik':ll,
              'logprior':lp,
              'logq':lq}

    return (1.-delta)*lq+delta*(ll+lp), auxout
    

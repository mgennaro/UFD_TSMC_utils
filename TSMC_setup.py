'''
Author: Mario Gennaro

This programs defines the setup for a particular
set of TSMC runs for the UFDs-optical analysis

Once this routine has saved the configurations,
all the other routines (TSMC_run, TSMC_completeness)
need to know is in which folder
the TSMC_setup.pickle file is stored

'''

import shelve, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KDTree
from synCMD.auxfunc.photband import photband
from synCMD.generators.GeneralRandom import GeneralRandom
import extinction
from scipy.io import readsav
from itertools import combinations

#################
#Paths were the setup file will be stored
##############

UFD = 'herc'
path = '/user/gennaro/UFDs_OPT/'
savepath='/user/gennaro/UFDs_OPT/'+UFD+'/TSMCout/'

#################
# Bands to use
#################

photbands = []  #List of photband objects

pb = photband()
pb.name = 'F606W'
pb.lowcut = 90.
pb.Alam_Av = (extinction.ccm89(np.array([5921.1]),1,3.1))[0]
photbands.append(pb)

pb = photband()
pb.name = 'F814W'
pb.lowcut  = 90.
pb.Alam_Av = (extinction.ccm89(np.array([8057.0]),1,3.1))[0]
photbands.append(pb)

#################
#Restore the isocrones and the NN interpolator list
#################

isoPD = pickle.load(open( '/user/gennaro/UFDs_OPT/shelves/isoACS.pickle', "rb" ) )
dfACSiso = isoPD['dfACSiso']
age_vals = isoPD['age_vals']
feh_vals = isoPD['feh_vals']
iso_NNInt = pickle.load(open('/user/gennaro/UFDs_OPT/shelves/iso_NNint.pickle',"rb"))

#################
#Artificial stars part
#################

#read the AS catalog
f2r = path+UFD+'/'+UFD+'.art'
dfAS = pd.read_table(f2r,header=None,sep='\s+',
                   names=['F606W_in','F606W_out','F814W_in','F814W_out'])

# Define the detection criterion (detflag=0:incomplete, detflag=1:complete)

outmags =  np.empty([len(photbands),len(dfAS)])
for i,pb in enumerate(photbands):
    outmags[i,:] = dfAS[pb.name+'_out'].values

msk = outmags <= 90

#################
#Define the limits of the domain for each paramter in the problem
#################

limits_all = [(np.log(0.4),np.log(5)),
              (np.amin(age_vals),np.amax(age_vals)),
              (np.amin(feh_vals),np.amax(feh_vals)),
              (20.64-1,20.64+1),
              (-1,1.),
              (0,1.),
              (0,1.)]

#################
# Create a list of generators objects of the same lenght
# as the number of MCMC parameters. These will
# be used to compute the q_TSMC probability (in this case uniform)
#################

# ln Mass 
mss_qGR = GeneralRandom(np.asarray([limits_all[0][0],limits_all[0][1]]),np.array([1,1]),1000)

#Age
age_qGR = GeneralRandom(np.asarray([limits_all[1][0],limits_all[1][1]]),np.array([1,1]),1000)
    
#Metallicity
feh_qGR = GeneralRandom(np.asarray([limits_all[2][0],limits_all[2][1]]),np.array([1,1]),1000)

#DM
DMT_qGR = GeneralRandom(np.asarray([limits_all[3][0],limits_all[3][1]]),np.array([1,1]),1000)

#Binary q
bin_qGR = GeneralRandom(np.asarray([limits_all[4][0],limits_all[4][1]]),np.array([1,1]),1000)

#A_606 - A_814
ex1_qGR = GeneralRandom(np.asarray([limits_all[5][0],limits_all[5][1]]),np.array([1,1]),1000)

#A_814
ex2_qGR = GeneralRandom(np.asarray([limits_all[6][0],limits_all[6][1]]),np.array([1,1]),1000)


qGRlist_all = [mss_qGR,age_qGR,feh_qGR,DMT_qGR,bin_qGR,ex1_qGR,ex2_qGR]

#################
# Create a list of generators objects of the same lenght
# as the number of MCMC parameters. These will
# be used to compute the prior probability
#################

# ln Mass (palpha is the linear alpha, i.e. alpha salpeter = -2.3)
palpha = -1.25
pml = np.linspace(limits_all[0][0],limits_all[0][1],200)
pmv = np.exp(pml*(palpha+1))
mss_GR = GeneralRandom(pml,pmv,1000)

#Age
sfh = pd.read_table('/user/gennaro/UFDs_OPT/'+UFD+'/'+UFD+'sfh.txt',header=None,sep='\s+',
                   names=['age','feh','weights'])
ages_sfh = np.unique(sfh.age.values)
marg_wgt = np.zeros_like(ages_sfh)
for i,aaa in enumerate(ages_sfh):
    marg_wgt[i] = np.sum(sfh[sfh.age == aaa].weights.values)
    
pal = np.linspace(limits_all[1][0],limits_all[1][1],250)
pav = np.zeros_like(pal)
for i,aaa in enumerate(ages_sfh):
    pav = pav + marg_wgt[i]*norm.pdf(pal,loc=aaa,scale=0.1)
pav = pav + 5 # adding a flat part
age_GR = GeneralRandom(pal,pav,1000)
    
#Metallicity
dicMDF = readsav('/user/gennaro/UFDs_OPT/MDFS/Summary_grid0p2_'+UFD+'_adp.sav')
pfl = feh_vals
pfv = dicMDF.mednmdf +0.4  # adding a flat part
feh_GR = GeneralRandom(pfl,pfv,1000)

#DM
DMT_GR = GeneralRandom(np.asarray([limits_all[3][0],limits_all[3][1]]),np.array([1,1]),1000)

#Binary q
pbl = np.array([-1,-1e-6,0.,1])
pbv = np.array([25,25,75,75])
bin_GR = GeneralRandom(pbl,pbv,1000)

#A_606 - A_814
ex1_GR = GeneralRandom(np.asarray([limits_all[5][0],limits_all[5][1]]),np.array([1,1]),1000)

#A_814
ex2_GR = GeneralRandom(np.asarray([limits_all[6][0],limits_all[6][1]]),np.array([1,1]),1000)


GRlist_all = [mss_GR,age_GR,feh_GR,DMT_GR,bin_GR,ex1_GR,ex2_GR]

#################
#Setup of the TSMC sampler 
#################

nwalkers = 10000
niters   = 10
ndims    = 4
nfact    = 15
tau      = 0.55

#################
# Default parameters & keys
#################

DM = 20.64
Av = 0.279
A_F606W = Av*photbands[0].Alam_Av
A_F814W = Av*photbands[1].Alam_Av
bq = -1

defaults = np.asarray([np.log(0.5),12,-2.5,DM,bq,A_F606W-A_F814W,A_F814W])
keys     = np.asarray([1,1,1,0,1,0,0],dtype=np.bool_)
cols     = ['ln M[Msun]','Age[Gyr]','[Fe/H][dex]','DM','bin q']
binsize  = [75,40,40,35,40]
for pb in photbands:
    cols.append(pb.name)
    binsize.append(25)
cols    = np.asarray(cols)[keys==True]
binsize = np.asarray(binsize)[keys==True]

#################
# Save the setup file
#################

TSMC_setup = {'photbands': photbands,
              'limits_all': limits_all,
              'qGRlist_all':qGRlist_all,
              'GRlist_all':GRlist_all,
              'cols':cols,
              'binsize':binsize,
              'nwalkers':nwalkers,
              'niters':niters,
              'ndims':ndims,
              'nfact':nfact,
              'tau':tau,
              'defaults':defaults,
              'keys':keys,
              'msk':msk,
              'UFD':UFD,
              'path':path
              }

pickle.dump( TSMC_setup, open(savepath+'Setup.pickle', "wb" ) )

# Author: Mario Gennaro

"""
This program reads the Oxigen-enhanced VanDenberg isochrones
computed by Tom Brown for his 2014 paper on UFDs (F606W, F814W)
and stores the whole grid as a pandas dataframe using shelve
"""

import pandas as pd
import shelve
import glob
import re
import numpy as np

#List all the iscrone files in the directory

flist = glob.glob('/user/gennaro/UFDs_OPT/isocrones/f*')

dfACSiso = pd.DataFrame()

for file in flist:
    print(file)
    metchar = (re.search('fm(.+?)a',file)).group(1)
    print(metchar)
    agechar = (re.search(metchar+'a(.+?)\.iso',file)).group(1)
    print(agechar)
    met = -1.*np.asarray(metchar,dtype=float)/100.
    age = np.asarray(agechar,dtype=float)/100.
    print(met,age)

    num_lines = sum(1 for line in open(file))

    arrays = [[np.repeat(met,num_lines)],[np.repeat(age,num_lines) ]]
    tuples = list(zip(*arrays))
    ind = pd.MultiIndex.from_tuples(tuples, names=['Z', 'age'])

    df1 = pd.read_table(file,header=None,sep='\s+',
          names=['mass','logT','logL','logg','F606W','F814W'])

        # df1.index
        #append zeta and age     to the dataframe
    dfACSiso = dfACSiso.append(df1)

dfACSiso.head()

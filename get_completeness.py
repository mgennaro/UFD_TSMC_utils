import numpy as np
from sklearn.neighbors import KDTree

def get_completeness(mags,AStree,dfAS,radius=0.02):
    '''query radius for AS arount input magnitude and then see how many
    are detected'''

    if(np.any(~np.isfinite(mags))):
        return 0,0
    else:
        ind   = AStree.query_radius(mags, r=radius) 
        neigh = len(ind[0])
        cnt   = np.sum(dfAS.iloc[ind[0]]['det_flag'].values,dtype=float)

        return cnt/neigh, neigh

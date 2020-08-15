import numpy as np
import argparse
import socket
import time
import os
import scipy.misc
import sys
import glob
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold, preprocessing


import os,sys,inspect
currentdir = "/usr/local/featureExtractionParty/external/pointnet_spine_ae"
sys.path.insert(0,currentdir) 
import importlib

import pandas as pd
from sklearn.cluster import KMeans

import meshparty
import time
from meshparty import trimesh_io, trimesh_vtk
import trimesh
from trimesh.primitives import Sphere
import os
import h5py
from meshparty import skeleton, skeleton_io
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from meshparty import mesh_filters
from meshparty import utils
import numpy as np
import pickle
import pandas
import cloudvolume
from meshparty import skeletonize
from trimesh.ray import ray_pyembree
from multiprocessing import Pool
from functools import partial
from scipy import sparse
from meshparty import skeleton_io
from annotationframeworkclient import infoservice
import trimesh
print("Finished imports")

# Import dually innervated spines list
spines_file = 'spines/dually_innervated_spines-nocutoff-unfinishedcopy.pkl'
spines_df = pd.read_pickle(spines_file)

# Import original dataframe
basil_synapses_file = 'Basil_synapses_cleaned_final.df'
data = pandas.read_csv(basil_synapses_file)

spine1_oldids = []
spine2_oldids = []
spine1_synapses = []
spine2_synapses = []
for i in spines_df.index:
    data_synapses = data.loc[data['postsyn_segid'] == int(spines_df['cell id'][i])]
    syn1_ind = 0
    for j in data_synapses.index:
        if (data_synapses['cleft_segid'][j] != int(spines_df['synapse 1 id'][i])):
            syn1_ind += 1
        else:
            break
    syn1 = data_synapses.cleft_segid.values[spines_df['spine 1 index'][i]]
    s1x = data_synapses.loc[data_synapses['cleft_segid']==syn1].centroid_x.values[0]
    s1y = data_synapses.loc[data_synapses['cleft_segid']==syn1].centroid_y.values[0]
    s1z = data_synapses.loc[data_synapses['cleft_segid']==syn1].centroid_z.values[0]
    s1 = list([s1x,s1y,s1z])
    spine1_oldids.append(syn1_ind)
    spine1_synapses.append(s1)

    syn2_ind = 0
    for k in data_synapses.index:
        if (data_synapses['cleft_segid'][k] != int(spines_df['synapse 2 id'][i])):
            syn2_ind += 1
        else:
            break
    syn2 = data_synapses.cleft_segid.values[spines_df['spine 2 index'][i]] 
    s2x = data_synapses.loc[data_synapses['cleft_segid']==syn2].centroid_x.values[0]
    s2y = data_synapses.loc[data_synapses['cleft_segid']==syn2].centroid_y.values[0]
    s2z = data_synapses.loc[data_synapses['cleft_segid']==syn2].centroid_z.values[0]
    s2 = list([s2x,s2y,s2z])
    spine2_oldids.append(syn2_ind)
    spine2_synapses.append(s2)
    print(i)

new_df = spines_df
new_df['spine 1 old index'] = spine1_oldids
new_df['spine 1 synapse'] = spine1_synapses
new_df['spine 2 old index'] = spine2_oldids
new_df['spine 2 synapse'] = spine2_synapses

outputfile = 'dually-innervated-unfinished_reindexed.pkl'
pickle.dump(new_df,open(outputfile, "wb" ))

check_csv = 'dually-innervated-unfinished_reindexed.csv'
check_file = open('dually-innervated-unfinished_reindexed.pkl',"rb")
check_db = pickle.load(check_file)
check_file.close()
check_df = pd.DataFrame(check_db)
check_df.to_csv(check_csv)
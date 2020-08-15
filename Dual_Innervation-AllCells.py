import tensorflow as tf
import numpy as np
import argparse
import socket
import time
import os
import scipy.misc
import sys
import glob
import umap
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold, preprocessing


import os,sys,inspect
currentdir = "/usr/local/featureExtractionParty/external/pointnet_spine_ae"
sys.path.insert(0,currentdir) 
import provider
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
import cgal_functions_Module as cfm
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
import pyembree
from trimesh.ray import ray_pyembree
from multiprocessing import Pool
from functools import partial
from scipy import sparse
from meshparty import skeleton_io
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from annotationframeworkclient import infoservice
from itkwidgets import view
import trimesh
print("Finished imports")

#GET ALL BASIL CELL IDs
import pandas as pd
filename = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/basil/analysis_dataframe/neurons_phenograph_cluster.pkl'
neuron_df = pd.read_pickle(filename)
neuron_df.shape
neuron_df['soma_id'].values
cell_id_list = neuron_df['soma_id']

# Create dataframe of information (including class) for all cells
classes_file = open(r'/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/basil/PSS/classification/synapses_PSS_backup.pkl',"rb")
classes_db = pickle.load(classes_file)
classes_file.close()
classes_df = pd.DataFrame(classes_db)
print(classes_df)

# Create dataframe of only spines for one cell
def return_spines(cell_id):
    cell_spines_df = pd.DataFrame()
    query = 'post_pt_root_id == ' + str(cell_id)
    cell_classes_df = classes_df.query(query, inplace = False)
    for i in cell_classes_df.index:
        if (cell_classes_df['class (linear SVC, 1024 features)'][i] == '1'):
            cell_spines_df = cell_spines_df.append(cell_classes_df.iloc[i], ignore_index=False, sort=False)
            cell_spines_df = cell_spines_df.reindex(cell_classes_df.columns, axis=1)
    return cell_spines_df

# Create a dataframe of overlapping spines for one cell
def return_overlapping_spines(cell_id):
    cell_spines_df = return_spines(cell_id)
    ind_count = 1
    cell_overlapping_spines = pd.DataFrame(columns=['cell id','spine 1 index','synapse 1 id','spine 2 index','synapse 2 id','number of shared vertices'])
    for i in cell_spines_df.index:
        spine1_file = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/' + cell_id + '/spine_' + str(i) + ".off"
        for j in cell_spines_df.index[ind_count::]:
            spine2_file = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/' + cell_id + '/spine_' + str(j) + ".off"
            spinemesh1 = trimesh.exchange.load.load_mesh(spine1_file)
            spinemesh2 = trimesh.exchange.load.load_mesh(spine2_file)
            shared = meshparty.mesh_filters.filter_spatial_distance_from_points(spinemesh1,spinemesh2.vertices,1)
            if (True in shared):
                num_overlap = 0
                for b in shared:
                    if (b == True):
                        num_overlap += 1
                overlap_pair = {'cell id':cell_id,
                               'spine 1 index':i, 'synapse 1 id':cell_spines_df['id'][i],
                               'spine 2 index':j, 'synapse 2 id':cell_spines_df['id'][j],
                               'number of shared vertices': num_overlap}
                cell_overlapping_spines = cell_overlapping_spines.append(overlap_pair,ignore_index=True)
        ind_count += 1
    return(cell_overlapping_spines)

# Save to output file
def save_output():
    outputfile = '/usr/local/featureExtractionParty/victoria_notebooks/dually_innervated_spines/dually_innervated_spines-nocutoff.pkl'
    pickle.dump(dual_innervation_full,open(outputfile, "wb" ))
    
    
# Run on multiple cells
dual_innervation_full = pd.DataFrame()
cell_number = 0
for cell_id in cell_id_list:
    print(str(cell_number) + ": " + str(cell_id))
    logfile = open(r'/usr/local/featureExtractionParty/victoria_notebooks/dually_innervated_spines/logfile.txt','a')
    logfile.write(str(cell_number) + ": " + str(cell_id))
    logfile.close()
    dual_innervation_full = dual_innervation_full.append(return_overlapping_spines(str(cell_id)),ignore_index=True)
    save_output()
    cell_number += 1
    
    
# Read data from selected embedded points file and create a dataframe
check_csv = r'/usr/local/featureExtractionParty/victoria_notebooks/dually_innervated_spines/dually_innervated_spines-nocutoff.csv'
check_file = open(r"/usr/local/featureExtractionParty/victoria_notebooks/dually_innervated_spines/dually_innervated_spines-nocutoff.pkl","rb")
check_db = pickle.load(check_file)
check_file.close()
check_df = pd.DataFrame(check_db)
check_df.to_csv(check_csv)
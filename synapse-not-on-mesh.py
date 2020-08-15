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
from meshparty import trimesh_io, trimesh_vtk, trimesh_repair
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
from meshparty import skeletonize
from meshparty import mesh_skel_utils
from scipy.sparse import csgraph
print("Finished imports")

# Import full classifications dataframe
classes_file = 'synapses_PSS.pkl'
classes_df = pd.read_pickle(classes_file)

# Import dually innervated spines list
spines_file = 'dually-innervated-unfinished_reindexed.pkl'
spines_df = pd.read_pickle(spines_file)

# Import results list
results_file = r'Testing/results-first150.csv'
results_df = pd.read_csv(results_file)
print(results_df)

# Load meshes and create mesh actors for each mesh
def loadmeshes(i):
    spine1_file = r'Meshes/' + str(spines_df['cell id'][i]) + '/' + str(i) + '/spine_' + str(spines_df['spine 1 index'][i]) + '.off'
    spinemesh1 = trimesh.exchange.load.load_mesh(spine1_file)
    spinemesh1_mp = meshparty.trimesh_io.Mesh(vertices=spinemesh1.vertices, faces=spinemesh1.faces)
    spinemesh1_actor = trimesh_vtk.mesh_actor(spinemesh1, color=(0,0,1), opacity=0.2)
    spine2_file = r'Meshes/' + str(spines_df['cell id'][i]) + '/' + str(i) + '/spine_' + str(spines_df['spine 2 index'][i]) + '.off'
    spinemesh2 = trimesh.exchange.load.load_mesh(spine2_file)
    spinemesh2_mp = meshparty.trimesh_io.Mesh(vertices=spinemesh2.vertices, faces=spinemesh2.faces)
    spinemesh2_actor = trimesh_vtk.mesh_actor(spinemesh2, color=(1,0,0), opacity=0.2)
    return spinemesh1_mp, spinemesh2_mp, spinemesh1_actor, spinemesh2_actor

# Find synapse location and make a point cloud actor for the synapse of each mesh
def plotsynapses(i):
    synapse1_loc = [spines_df['spine 1 synapse'][i][0]*4,spines_df['spine 1 synapse'][i][1]*4,spines_df['spine 1 synapse'][i][2]*40]
    synapse2_loc = [spines_df['spine 2 synapse'][i][0]*4,spines_df['spine 2 synapse'][i][1]*4,spines_df['spine 2 synapse'][i][2]*40]
    synapse1_actor = trimesh_vtk.point_cloud_actor([synapse1_loc],size=100,color=(0,0,1),opacity=1)
    synapse2_actor = trimesh_vtk.point_cloud_actor([synapse2_loc],size=100,color=(1,0,0),opacity=1)
    return synapse1_loc, synapse2_loc, synapse1_actor, synapse2_actor

# Skeletonize and create a skeleton actor for each mesh
def skeletonizemesh():
    skeleton1 = meshparty.skeletonize.skeletonize_mesh(spinemesh1_mp,compute_radius=False,cc_vertex_thresh=0)
    skeleton1_actor = meshparty.trimesh_vtk.skeleton_actor(skeleton1,color=(0,0,1),opacity=1)
    skeleton2 = meshparty.skeletonize.skeletonize_mesh(spinemesh2_mp,compute_radius=False,cc_vertex_thresh=0)
    skeleton2_actor = meshparty.trimesh_vtk.skeleton_actor(skeleton2,color=(1,0,0),opacity=1)
    return skeleton1, skeleton2, skeleton1_actor, skeleton2_actor

# Determine whether synapse is on the mesh
def synapse_not_on_mesh(spinemesh1_mp, spinemesh2_mp):
    compare1 = meshparty.mesh_filters.filter_spatial_distance_from_points(spinemesh1_mp,synapse1_loc,120)
    if True not in compare1:
        print("blue")
        return True
    compare2 = meshparty.mesh_filters.filter_spatial_distance_from_points(spinemesh2_mp,synapse2_loc,120)
    if True not in compare2:
        print("red")
        return True

far_list = []
for i in range(results_df.shape[0]):
    spinemesh1_mp, spinemesh2_mp, spinemesh1_actor, spinemesh2_actor = loadmeshes(i)
    synapse1_loc, synapse2_loc, synapse1_actor, synapse2_actor = plotsynapses(i)
    skeleton1, skeleton2, skeleton1_actor, skeleton2_actor = skeletonizemesh()
    if synapse_not_on_mesh(spinemesh1_mp, spinemesh2_mp) == True:
        print(str(i) + ": synapse not on mesh")
        far_list.append(i)
print(far_list)

# Visualize the meshes, synapses, and lines (no skeletons)
#renderer1 = trimesh_vtk.render_actors(actors=[spinemesh1_actor,spinemesh2_actor,synapse1_actor,synapse2_actor],do_save=False)
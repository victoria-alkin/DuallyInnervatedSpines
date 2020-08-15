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
spines_file = r'dually-innervated_reindexed.pkl'
spines_df = pd.read_pickle(spines_file)

# Choose which pair of meshes to work with (i corresponds to index in list of pairs)
i = 39344

# Load meshes and create mesh actors for each mesh
spine1_file = r'Meshes/' + str(spines_df['cell id'][i]) + '/' + str(i) + '/spine_' + str(spines_df['spine 1 index'][i]) + '.off'
spinemesh1 = trimesh.exchange.load.load_mesh(spine1_file)
spinemesh1_mp = meshparty.trimesh_io.Mesh(vertices=spinemesh1.vertices, faces=spinemesh1.faces)
spinemesh1_actor = trimesh_vtk.mesh_actor(spinemesh1, color=(0,0,1), opacity=0.2)
spine2_file = r'Meshes/' + str(spines_df['cell id'][i]) + '/' + str(i) + '/spine_' + str(spines_df['spine 2 index'][i]) + '.off'
spinemesh2 = trimesh.exchange.load.load_mesh(spine2_file)
spinemesh2_mp = meshparty.trimesh_io.Mesh(vertices=spinemesh2.vertices, faces=spinemesh2.faces)
spinemesh2_actor = trimesh_vtk.mesh_actor(spinemesh2, color=(1,0,0), opacity=0.2)

# Find synapse location and make a point cloud actor for the synapse of each mesh
synapse1_loc = [spines_df['spine 1 synapse'][i][0]*4,spines_df['spine 1 synapse'][i][1]*4,spines_df['spine 1 synapse'][i][2]*40]
synapse2_loc = [spines_df['spine 2 synapse'][i][0]*4,spines_df['spine 2 synapse'][i][1]*4,spines_df['spine 2 synapse'][i][2]*40]
synapse1_actor = trimesh_vtk.point_cloud_actor([synapse1_loc],size=100,color=(0,0,1),opacity=1)
synapse2_actor = trimesh_vtk.point_cloud_actor([synapse2_loc],size=100,color=(1,0,0),opacity=1)

# Find opening for each mesh
spine1broken = trimesh.repair.broken_faces(spinemesh1_mp)
spine1openings = spinemesh1_mp.submesh([spine1broken], append=True)
spine1openings = meshparty.trimesh_io.Mesh(vertices=spine1openings.vertices, faces=spine1openings.faces)
print(spine1openings.body_count)
if spine1openings.body_count == 1:
    print("PSS 1: spine")
elif spine1openings.body_count > 3:
    print("PSS 1: shaft")
else:
    mask1 = mesh_filters.filter_components_by_size(spine1openings,min_size=10)
    if True not in mask1:
        print("PSS 1: spine")
    else:
        spine1openings = spine1openings.apply_mask(mask1)
        print(spine1openings.body_count)
        if spine1openings.body_count == 1:
            print("PSS 1: spine")
        else:
            print("PSS 1: shaft")

spine2broken = trimesh.repair.broken_faces(spinemesh2_mp)
spine2openings = spinemesh2_mp.submesh([spine2broken], append=True)
spine2openings = meshparty.trimesh_io.Mesh(vertices=spine2openings.vertices, faces=spine2openings.faces)
print(spine2openings.body_count)
if spine2openings.body_count == 1:
    print("PSS 2: spine")
elif spine2openings.body_count > 3:
    print("PSS 2: shaft")
else:
    mask2 = mesh_filters.filter_components_by_size(spine2openings,min_size=10)
    if True not in mask2:
        print("PSS 2: spine")
    else:
        spine2openings = spine2openings.apply_mask(mask2)
        print(spine2openings.body_count)
        if spine2openings.body_count == 1:
            print("PSS 2: spine")
        else:
            print("PSS 2: shaft")

# Determine whether at least one of the two pss is a shaft
def shaft():
    spine1broken = trimesh.repair.broken_faces(spinemesh1_mp)
    spine1openings = spinemesh1_mp.submesh([spine1broken], append=True)
    spine1openings = meshparty.trimesh_io.Mesh(vertices=spine1openings.vertices, faces=spine1openings.faces)
    if spine1openings.body_count == 1:
        pass
    elif spine1openings.body_count > 3:
        return True
    else:
        mask1 = mesh_filters.filter_components_by_size(spine1openings,min_size=10)
        if True not in mask1:
            pass
        else:
            spine1openings = spine1openings.apply_mask(mask1)
            print(spine1openings.body_count)
            if spine1openings.body_count == 1:
                pass
            else:
                return True

    spine2broken = trimesh.repair.broken_faces(spinemesh2_mp)
    spine2openings = spinemesh2_mp.submesh([spine2broken], append=True)
    spine2openings = meshparty.trimesh_io.Mesh(vertices=spine2openings.vertices, faces=spine2openings.faces)
    print(spine2openings.body_count)
    if spine2openings.body_count == 1:
        pass
    elif spine2openings.body_count > 3:
        return True
    else:
        mask2 = mesh_filters.filter_components_by_size(spine2openings,min_size=10)
        if True not in mask2:
            pass
        else:
            spine2openings = spine2openings.apply_mask(mask2)
            print(spine2openings.body_count)
            if spine2openings.body_count == 1:
                pass
            else:
                return True
    return False

# Visualize the meshes, synapses, and lines (no skeletons)
renderer1 = trimesh_vtk.render_actors(actors=[spinemesh1_actor,spinemesh2_actor,synapse1_actor,synapse2_actor],do_save=False)
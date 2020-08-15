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
        return True
    compare2 = meshparty.mesh_filters.filter_spatial_distance_from_points(spinemesh2_mp,synapse2_loc,120)
    if True not in compare2:
        return True

# Determine whether the two pss are adjacent spines
def adjacentspines(i):
    if spines_df['number of shared vertices'][i] <= 50:
        return True
    else:
        return False

# Determine whether the synapses are close enough together that it is definitely a dually innervated spine
def synapse_dist_small(synapse1_loc,synapse2_loc):
    syn1loc_array = np.array(synapse1_loc)
    syn2loc_array = np.array(synapse2_loc)
    squared_dist = np.sum((syn1loc_array-syn2loc_array)**2,axis=0)
    syn_dist = np.sqrt(squared_dist)
    print(syn_dist)
    if syn_dist < 550:
        return True

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

# Determine whether it is a double-headed spine (with an innervation on each head)
def doubleheaded():
    # Find opening and point on skeleton closest to opening for each mesh
    spine1broken = trimesh.repair.broken_faces(spinemesh1_mp)
    spine1openings = spinemesh1_mp.submesh([spine1broken], append=True)
    spine1openings = meshparty.trimesh_io.Mesh(vertices=spine1openings.vertices, faces=spine1openings.faces)
    print(spine1openings)
    if spine1openings.body_count == 1:
        path_opening1_a = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh1_mp,skeleton1,spine1openings.vertices[0])
        opening_ind1_a = np.where(skeleton1.vertices == spinemesh1_mp.vertices[path_opening1_a[-1]])
        opening_ind1_a = opening_ind1_a[0][0]
        opening_ind1_b = -1
    elif spine1openings.body_count == 2:
        mask1 = mesh_filters.filter_largest_component(spine1openings)
        spine1openings_a = spine1openings.apply_mask(mask1)
        spine1openings_b_vertices = []
        for i in spine1openings.vertices:
            if i not in spine1openings_a.vertices:
                spine1openings_b_vertices.append(i.tolist())
        path_opening1_a = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh1_mp,skeleton1,spine1openings.vertices[0])
        path_opening1_b = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh1_mp,skeleton1,spine1openings_b_vertices[0])
        opening_ind1_a = np.where(skeleton1.vertices == spinemesh1_mp.vertices[path_opening1_a[-1]])
        opening_ind1_a = opening_ind1_a[0][0]
        opening_ind1_b = np.where(skeleton1.vertices == spinemesh1_mp.vertices[path_opening1_b[-1]])
        opening_ind1_b = opening_ind1_b[0][0]
    else:
        mask1 = mesh_filters.filter_largest_component(spine1openings)
        spine1openings = spine1openings.apply_mask(mask1)
        path_opening1_a = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh1_mp,skeleton1,spine1openings.vertices[0])
        opening_ind1_a = np.where(skeleton1.vertices == spinemesh1_mp.vertices[path_opening1_a[-1]])
        opening_ind1_a = opening_ind1_a[0][0]
        opening_ind1_b = -1

    spine2broken = trimesh.repair.broken_faces(spinemesh2_mp)
    spine2openings = spinemesh2_mp.submesh([spine2broken], append=True)
    spine2openings = meshparty.trimesh_io.Mesh(vertices=spine2openings.vertices, faces=spine2openings.faces)
    print(spine2openings)
    if spine2openings.body_count == 1:
        path_opening2_a = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh2_mp,skeleton2,spine2openings.vertices[0])
        opening_ind2_a = np.where(skeleton2.vertices == spinemesh2_mp.vertices[path_opening2_a[-1]])
        opening_ind2_a = opening_ind2_a[0][0]
        opening_ind2_b = -1
    elif spine2openings.body_count == 2:
        mask2 = mesh_filters.filter_largest_component(spine2openings)
        spine2openings_a = spine2openings.apply_mask(mask2)
        spine2openings_b_vertices = []
        for i in spine2openings.vertices:
            if i not in spine2openings_a.vertices:
                spine2openings_b_vertices.append(i.tolist())
        path_opening2_a = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh2_mp,skeleton2,spine2openings.vertices[0])
        path_opening2_b = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh2_mp,skeleton2,spine2openings_b_vertices[0])
        opening_ind2_a = np.where(skeleton2.vertices == spinemesh2_mp.vertices[path_opening2_a[-1]])
        opening_ind2_a = opening_ind2_a[0][0]
        opening_ind2_b = np.where(skeleton2.vertices == spinemesh2_mp.vertices[path_opening2_b[-1]])
        opening_ind2_b = opening_ind2_b[0][0]
    else:
        mask2 = mesh_filters.filter_largest_component(spine2openings)
        spine2openings = spine2openings.apply_mask(mask2)
        path_opening2_a = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh2_mp,skeleton2,spine2openings.vertices[0])
        opening_ind2_a = np.where(skeleton2.vertices == spinemesh2_mp.vertices[path_opening2_a[-1]])
        opening_ind2_a = opening_ind2_a[0][0]
        opening_ind2_b = -1

    # Create a path from the synapse to the opening for each mesh/synapse
    # The path consists of two parts: (1) a path from the synapse to the nearest point on the skeleton, and
    #                                 (2) a path along the skeleton to the opening
    path1_1 = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh1_mp,skeleton1,synapse1_loc)
    meshverts1 = spinemesh1_mp.vertices.tolist()
    skelverts1 = skeleton1.vertices.tolist()
    inds1_a = path1_1
    inds1_b = path1_1[1:]
    index = np.where(skeleton1.vertices == spinemesh1_mp.vertices[path1_1[-1]])
    index = index[0][0]
    if opening_ind1_b == -1:
        opening_ind1 = opening_ind1_a
        skeleton1.reroot(opening_ind1_a)
        path1_2 = skeleton1.path_to_root(index)
        for i in path1_2[1:]:
            mesh_ind = np.where(spinemesh1_mp.vertices == skeleton1.vertices[i])
            mesh_ind = mesh_ind[0][0]
            inds1_a.append(mesh_ind)
            inds1_b.append(mesh_ind)
        inds1_b = np.append(inds1_b,inds1_b[-1])
        line1 = meshparty.trimesh_vtk.linked_point_actor(np.array(meshverts1),np.array(meshverts1),inds_a=inds1_a,inds_b=inds1_b,opacity=1)
    else:
        skeleton1_a = skeleton1
        skeleton1_a.reroot(opening_ind1_a)
        path1_2a = skeleton1_a.path_to_root(index)
        skeleton1_b = skeleton1
        skeleton1_b.reroot(opening_ind1_b)
        path1_2b = skeleton1_b.path_to_root(index)
        if len(path1_2a) > len(path1_2b) and len(path1_2b) > 1:
            path1_2 = path1_2b
            opening_ind1 = opening_ind1_b
        else:
            path1_2 = path1_2a
            opening_ind1 = opening_ind1_a
        for i in path1_2[1:]:
            mesh_ind = np.where(spinemesh1_mp.vertices == skeleton1.vertices[i])
            mesh_ind = mesh_ind[0][0]
            inds1_a.append(mesh_ind)
            inds1_b.append(mesh_ind)
        inds1_b = np.append(inds1_b,inds1_b[-1])
        line1 = meshparty.trimesh_vtk.linked_point_actor(np.array(meshverts1),np.array(meshverts1),inds_a=inds1_a,inds_b=inds1_b,opacity=1)

    path2_1 = meshparty.mesh_skel_utils.point_to_skel_meshpath(spinemesh2_mp,skeleton2,synapse2_loc)
    meshverts2 = spinemesh2_mp.vertices.tolist()
    skelverts2 = skeleton2.vertices.tolist()
    inds2_a = path2_1
    inds2_b = path2_1[1:]
    index = np.where(skeleton2.vertices == spinemesh2_mp.vertices[path2_1[-1]])
    index = index[0][0]
    if opening_ind2_b == -1:
        opening_ind2 = opening_ind2_a
        skeleton2.reroot(opening_ind2_a)
        path2_2 = skeleton2.path_to_root(index)
        for i in path2_2[1:]:
            mesh_ind = np.where(spinemesh2_mp.vertices == skeleton2.vertices[i])
            mesh_ind = mesh_ind[0][0]
            inds2_a.append(mesh_ind)
            inds2_b.append(mesh_ind)
        inds2_b = np.append(inds2_b,inds2_b[-1])
        line2 = meshparty.trimesh_vtk.linked_point_actor(np.array(meshverts2),np.array(meshverts2),inds_a=inds2_a,inds_b=inds2_b,opacity=1)
    else:
        skeleton2_a = skeleton2
        skeleton2_a.reroot(opening_ind2_a)
        path2_2a = skeleton2_a.path_to_root(index)
        skeleton2_b = skeleton2
        skeleton2_b.reroot(opening_ind2_b)
        path2_2b = skeleton2_b.path_to_root(index)
        if len(path2_2a) > len(path2_2b) and len(path2_2b) > 1:
            path2_2 = path2_2b
            opening_ind2 = opening_ind2_b
        else:
            path2_2 = path2_2a
            opening_ind2 = opening_ind2_a
        for i in path2_2[1:]:
            mesh_ind = np.where(spinemesh2_mp.vertices == skeleton2.vertices[i])
            mesh_ind = mesh_ind[0][0]
            inds2_a.append(mesh_ind)
            inds2_b.append(mesh_ind)
        inds2_b = np.append(inds2_b,inds2_b[-1])
        line2 = meshparty.trimesh_vtk.linked_point_actor(np.array(meshverts2),np.array(meshverts2),inds_a=inds2_a,inds_b=inds2_b,opacity=1)

    end_actor1 = trimesh_vtk.point_cloud_actor([skelverts1[opening_ind1]],size=100,color=(0.7,0.9,1),opacity=1)
    end_actor2 = trimesh_vtk.point_cloud_actor([skelverts2[opening_ind2]],size=100,color=(0.9,0.7,0.6),opacity=1)

    # Compare the two lines
    line1points = []
    verts_ind1 = 0
    for i in meshverts1:
        if verts_ind1 in inds1_a:
            line1points.append(i)
        verts_ind1 += 1
    line2points = []
    verts_ind2 = 0
    for i in meshverts2:
        if verts_ind2 in inds2_a:
            line2points.append(i)
        verts_ind2 += 1
    class Linemesh:
        def __init__(self, vertices):
            self.vertices = vertices
    line1mesh = Linemesh(line1points)
    #cloud = trimesh.PointCloud(line1points)
    #line1mesh = cloud.convex_hull
    compare = meshparty.mesh_filters.filter_spatial_distance_from_points(line1mesh,line2points,200)
    print(compare)
    numTrue = 0
    for i in compare:
        if i == True:
            numTrue += 1
    if numTrue > len(compare)/2:
        return False
    else:
        return True


sortlist = []
sort_df = results_df
for i in range(results_df.shape[0]):
    spinemesh1_mp, spinemesh2_mp, spinemesh1_actor, spinemesh2_actor = loadmeshes(i)
    synapse1_loc, synapse2_loc, synapse1_actor, synapse2_actor = plotsynapses(i)
    skeleton1, skeleton2, skeleton1_actor, skeleton2_actor = skeletonizemesh()
    if synapse_not_on_mesh(spinemesh1_mp, spinemesh2_mp) == True:
        print(str(i) + ": synapse not on mesh")
        sortlist.append(-1)
    elif adjacentspines(i) == True:
        print(str(i) + ": adjacent spines")
        sortlist.append(0)
    elif shaft() == True:
        print(str(i) + ": shaft")
        sortlist.append(1)
    elif synapse_dist_small(synapse1_loc,synapse2_loc) == True:
        print(str(i) + ": dually innervated spine")
        sortlist.append(3)
    elif doubleheaded() == True:
        print(str(i) + ": double-headed spine")
        sortlist.append(2)
    else:
        print(str(i) + ": dually innervated spine")
        sortlist.append(3)
print(sortlist)
print(len(sortlist))
sort_df['sort 1'] = sortlist
sort_df.to_csv(r'Testing/results-first150_sort7.csv')
#renderer1 = trimesh_vtk.render_actors(actors=[spinemesh1_actor,spinemesh2_actor,synapse1_actor,synapse2_actor],do_save=False)
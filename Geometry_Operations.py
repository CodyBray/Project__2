#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:25:01 2021

@author: kendrick shepherd
"""

import math
import numpy as np
import sys

# length of the beam
def Length(bar):
    bar_node=bar.init_node
    bar_vector = BarNodeToVector (bar_node, bar)
    barlength = VectorTwoNorm(bar_vector)
    return barlength

# Find two norm (magnitude) of a vector
def VectorTwoNorm(vector):
    norm=0
    for i in range(0,len(vector)):
        norm += vector[i]**2
    return np.sqrt(norm)

# Find a shared node between two bars
def FindSharedNode(bar_1,bar_2):
    nodes_1=[bar_1.node_1, bar_1.node2]
    nodes_2 = [bar_2.node_1, bar_2.node_2]
    for node in nodes_1:
        if node in nodes_2:
            return node

# Given a bar and a node on that bar, find the other node
def FindOtherNode(node,bar):
    if(bar.init_node ==node):
        return bar.end_node
    elif node ==bar.end_node:
        return bar.init_node
    else:
        sys.exit("The input node is not on the bar")
# Find a vector from input node (of the input bar) in the direction of the bar
def BarNodeToVector(origin_node,bar):
    other_node = FindOtherNode(origin_node,bar)
    origin_loc=origin_node.location
    other_loc = other_node.location
    vector = tuple(other_loc[i]-origin_loc[i]for i in range(len(origin_loc)))
    return vector

# Convert to bars that meet at a node into vectors pointing away from that node
def BarsToVectors(bar_1,bar_2):
    a1, b1 = bar_1
    a2, b2 = bar_2

    # find the common node
    if a1 == a2:
        node = a1
        v1_end, v2_end = b1, b2
    elif a1 == b2:
        node = a1
        v1_end, v2_end = b1, a2
    elif b1 == a2:
        node = b1
        v1_end, v2_end = a1, b2
    elif b1 == b2:
        node = b1
        v1_end, v2_end = a1, a2
    else:
        raise ValueError("Bars do not meet at a common node")

    # create vectors pointing away from the node
    v1 = tuple(v1_end[i] - node[i] for i in range(len(node)))
    v2 = tuple(v2_end[i] - node[i] for i in range(len(node)))

    return v1, v2

# Cross product of two vectors
def TwoDCrossProduct(vec1,vec2):
    cross_product = vec1[0]*vec2[1]-vec1[1]*vec2[0]
    return cross_product

# Dot product of two vectors
def DotProduct(vec1,vec2):
    Dproduct=np.dot(vec1,vec2)
    return Dproduct

# Cosine of angle from local x vector direction to other vector
def CosineVectors(local_x_vec,other_vec):
    dotproduct = DotProduct(local_x_vec,other_vec)
    norm_x = VectorTwoNorm(local_x_vec)
    norm_W = VectorTwoNorm(other_vec)
    cos_theta= dotproduct/ (norm_x*norm_W)
    return cos_theta

# Sine of angle from local x vector direction to other vector
def SineVectors(local_x_vec,other_vec):
    cross_product = TwoDCrossProduct(local_x_vec,other_vec)
    norm_x = VectorTwoNorm(local_x_vec)
    norm_W = VectorTwoNorm(other_vec)
    sin_theta= cross_product/ (norm_x*norm_W)
    return sin_theta

# Cosine of angle from local x bar to the other bar
def CosineBars(local_x_bar,other_bar):
    origin_node = FindSharedNode(local_x_bar, other_bar)

    local_x_vec = BarNodeToVector(origin_node, local_x_bar)
    other_vec   = BarNodeToVector(origin_node, other_bar)

    return CosineVectors(local_x_vec, other_vec)

# Sine of angle from local x bar to the other bar
def SineBars(local_x_bar,other_bar):
    origin_node = FindSharedNode(local_x_bar, other_bar)

    local_x_vec = BarNodeToVector(origin_node, local_x_bar)
    other_vec   = BarNodeToVector(origin_node, other_bar)

    return SineVectors(local_x_vec, other_vec)
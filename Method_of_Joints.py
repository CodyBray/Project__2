#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:37:32 2021

@author: kendrick shepherd
"""

import sys

import Geometry_Operations as geom

# Determine the unknown bars next to this node


def UnknownBars(node):
    for node in node:
        unknown_bars = [
            bar for bar in node.bars
            if not bar.is_computed
            ]
    return len(unknown_bars(node))


# Determine if a node if "viable" or not
def NodeIsViable(node): 
   if len(unknown_bars)<=2:
      return rue
   else:
      return false
   for node in node:
       unknown_bars = [
           bar for bar in node.bars
           if not bar.is_computed
           ]
       if len(unknown_bars) <= 2 :
        return True
   else:
        return False
    
# Compute unknown force in bar due to sum of the
# forces in the x direction
def SumOfForcesInLocalX(node, local_x_bar):
    local_x_
    totalsumforce=0
    local_x_bar = unknown_bars[0]
    global_force_vec = (node.GetNetXFroce(),node.GetNetYForce())
    
    return

# Compute unknown force in bar due to sum of the 
# forces in the y direction
def SumOfForcesInLocalY(node, unknown_bars):
    return
    
# Perform the method of joints on the structure
def IterateUsingMethodOfJoints(nodes,bars):
    return
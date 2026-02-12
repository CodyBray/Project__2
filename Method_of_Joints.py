import sys

import Geometry_Operations as geom

# Determine the unknown bars next to this node


def UnknownBars(node):
    for node in node:
        unknown_bars = [
            bar for bar in node.bars
            if not bar.is_computed
            ]
    return len(unknown_bars)

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
    local_x_Vec = geom.BarNodeToVector(node,local_x_bar)
    xforce=0
    net_fx = node.GetNetXForce()
    net_fy = node.GetNetYForce()
    
    xforce += net_fx * geom.CosineVectors(local_x_Vec, [1,0])
    xforce += net_fy * geom.CosineVectors(local_x_Vec, [0,1])
    
    
    for bar in node.bars:
        if bar.is_computed:
            vec_this_bar = geom.BarNodeToVector(node, bar)
            xforce += net_fx * geom.CosineVectors(local_x_Vec,vec_this_bar)
    local_x_bar.axial_load=-xforce
    local_x_bar.is_computed = True
    return
# Compute unknown force in bar due to sum of the 
# forces in the y direction
def SumOfForcesInLocalY(node, unknown_bars):

    local_x_bar = unknown_bars[0]
    local_x_Vec = geom.BarNodeToVector(node,local_x_bar)
    
    totalsumforce=0
    net_fx = node.GetNetXForce()
    net_fy = node.GetNetYForce()
    
    totalsumforce += net_fx * geom.SineVectors(local_x_Vec, [1,0])
    totalsumforce += net_fy * geom.SineVectors(local_x_Vec, [0,1])
    
    
    for bar in node.bars:
        if bar.is_computed:
            vec_this_bar = geom.BarNodeToVector(node, bar)
            totalsumforce += net_fx * geom.SineVectors(local_x_Vec,vec_this_bar)

# Perform the method of joints on the structure
def IterateUsingMethodOfJoints(nodes,bars):
    progress = True
    while progress:
        progress = False
        
        for node in nodes:
            Unknown_bars = [
                bar for bar in node.bars
                if not bar.is_computed
                ]
            if len(Unknown_bars) == 1:
               SumOfForcesInLocalX(node, Unknown_bars[0])
               progress = True
            elif len(Unknown_bars) == 2:
               SumOfForcesInLocalY(node, Unknown_bars)
               SumOfForcesInLocalX(node, Unknown_bars[0])
               progress = True
   # After loop ends, check if all bars solved
    for bar in bars:
       if not bar.is_computed:
           raise ValueError("Structure not fully solvable.")
            
    return

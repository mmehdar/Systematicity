# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:26:21 2022

@author: mmehd

calculate role distance for each role/candidate

"""
import numpy as np



def roleDistance(
        role_skill_pos,role_skill_val,agents_skill_pos,agents_skill_val, 
        training_formal_pos,training_formal_val, 
        training_informal_pos,training_informal_val
        ):
    
    #probably ineffecient but fast to code with new matrices
    rsp = role_skill_pos[:,np.newaxis,:]
    asp = agents_skill_pos[:,:,np.newaxis]
    tfp = training_formal_pos[:,:,np.newaxis]
    tip = training_informal_pos[:,:,np.newaxis]

    rsv = role_skill_val[:,np.newaxis,:]
    asv = agents_skill_val[:,:,np.newaxis]
    tfv = training_formal_val[:,:,np.newaxis]
    tiv = training_informal_val[:,:,np.newaxis]
        
    
    #IF DISTANCE IS ZERO, MULTIPLIER IS ONE OTHERWISE LESS
    #
    agent_dist = (1-np.absolute(rsp-asp)) * (
        1-rsv+asv)     
    formal_dist = (1-np.absolute(rsp-tfp)) * (
        1-rsv+tfv)
    informal_dist = (1-np.absolute(rsp-tip)) * (
        1-rsv+tiv)
    return agent_dist +formal_dist+informal_dist


         
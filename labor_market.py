# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 20:14:50 2022

@author: mmehdar

this script creates a simplified labor market model with skills lying on a 
2-d plane, companies do not see the underlying skils but rely on sigal features
instead (such as degree/similarity of previous role). The main driving dynamics 
are: 1- the information is noisy/uncertain 2- there are applicants in majority/
minority groups with the minority having less access to formal signals 3- over 
time the minority gets more access to informal training on skills but the 
pretrained company models do not recognize those which leads to higher search 
times and ineffeciancies measured by selecting less optimal candidates 
"""

import numpy as np
from roleDistance import roleDistance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


#set seed for reproducability
np.random.seed(1)


#parameters
nagents = 1000
nroles = 500

pminority = np.array([0.0,0.2,0.4,0.6,0.8,1])
pmajority = 1-pminority
#this is a parameter for the training gap
gap = 0.5


#1 is first dimeinsion position e.g. quantitative, 2 is 2d position, 
agents_skill_pos = np.random.uniform(size = [2,nagents])
agents_skill_val = np.random.uniform(size = [2,nagents])

#indicator if currently employed - NOT USED YET
#agents_employed = np.array(1,nagents)


#similar to agents, roles have positions and values
role_skill_pos = np.random.uniform(size=[2,nroles])
role_skill_val = np.random.uniform(size=[2,nroles])

score_gap = np.zeros(len(pminority))

for i in range(len(pminority)):
    

    # indicator majority = 1 minority = 0
    agents_type = np.random.choice(a = [0,1],p = [pminority[i],pmajority[i]], 
                                   size = [1,nagents])
     
    #first step: initialize skills based on training : care to make overall formal 
    #and informal equal in aggregate
    
    training_formal_pos = np.random.uniform(size = [2,nagents])
    training_informal_pos = np.random.uniform(size = [2,nagents])
    
    #here we weight the training so more formal for majority, informal minority
    training_formal_val = np.random.uniform(size = [2,nagents])
    training_informal_val = np.random.uniform(size = [2,nagents])

    training_formal_val = (training_formal_val + 
                           training_formal_val*(1+gap)*agents_type)
    training_informal_val = (training_informal_val + 
                             training_informal_val*(1+gap)*(1-agents_type))
    
    
    role_distance = roleDistance(role_skill_pos,role_skill_val,agents_skill_pos,
                                 agents_skill_val, 
                                 training_formal_pos,training_formal_val, 
                                 training_informal_pos,training_informal_val)
    
    role_distance_1d = np.sum(role_distance, axis=0)
    best_score = np.max(role_distance_1d,axis = 0)
    best_candidate = np.argmax(role_distance_1d,axis = 0)
    
    #don't take informal into account
    mrole_distance = roleDistance(role_skill_pos,role_skill_val,agents_skill_pos,
                                 agents_skill_val, 
                                 training_formal_pos,training_formal_val, 
                                 training_informal_pos,training_informal_val*0)
    
    
    mrole_distance_1d = np.sum(mrole_distance, axis=0)
    mbest_score = np.max(mrole_distance_1d,axis = 0)
    mbest_candidate = np.argmax(mrole_distance_1d,axis = 0)
    
    #I convert scores to the full model to compare both 
    I = np.indices(mbest_candidate.shape)
    mbest_score_adjusted = np.squeeze(role_distance_1d[mbest_candidate,I])
    
    #I check that this works correctly 
    mbest_score_check = mrole_distance_1d[mbest_candidate,I]
    print("check that sum is zero " + str(np.sum(mbest_score-mbest_score_check)))
    
    ax = sns.kdeplot(best_score)
    sns.kdeplot(mbest_score_adjusted, ax=ax)
    plt.legend(labels=['full', 'formal'], title="scores at p-minority = "+str(pminority[i]))
    plt.show()
    
    score_gap[i] = (np.sum(best_score)-np.sum(mbest_score_adjusted))/np.sum(mbest_score_adjusted)
    
    

ax2 = plt.plot(pminority,score_gap)
plt.title("score gap by percentage of informal trained in labor force")

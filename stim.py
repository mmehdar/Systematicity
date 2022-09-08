# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# import required modules
#import pandas
import pandas as pd
import random
# import the metrics class
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 


scale=StandardScaler()



# load dataset
pima = pd.read_csv("data.csv")
pima.dtypes


#creaate dummies for application status
pima = pd.get_dummies(pima, columns = ['application_status'])


pima['co_1'] = pima.candidate_skill_1_count * pima.occupation_skill_1_count 
pima['co_2'] = pima.candidate_skill_2_count * pima.occupation_skill_2_count 
pima['co_3'] = pima.candidate_skill_3_count * pima.occupation_skill_3_count 
pima['co_4'] = pima.candidate_skill_4_count * pima.occupation_skill_4_count 
pima['co_5'] = pima.candidate_skill_5_count * pima.occupation_skill_5_count 
pima['co_6'] = pima.candidate_skill_6_count * pima.occupation_skill_6_count 
pima['co_7'] = pima.candidate_skill_7_count *  pima.occupation_skill_7_count 
pima['co_8'] = pima.candidate_skill_8_count * pima.occupation_skill_8_count 
pima['co_9'] = pima.candidate_skill_9_count * pima.occupation_skill_9_count 




pima.replace([np.inf, -np.inf], np.nan, inplace=True)   
pima.fillna(pima.mean(), inplace = True )
#pima.fillna(0, inplace = True )

#here I use the bootstrap method to create different models
#values = pima.values

#Lets configure Bootstrap

n_iterations = 5  #No. of bootstrap samples to be repeated (created)
n_size = int(len(pima) * 1) #Size of sample, picking  100% of the given data in every bootstrap sample

   

#Lets run Bootstrap
stats = list()
logreg_array = []
y_test_array = []
y_pred_array = []



#default model
#features = [1] + list(range(13,15)) + list(range(29,59))  + list(range(62,71))
#features = [1] + list(range(62,71))

features = [1] + list(range(4,15)) + list(range(16,20)) + list(range(23,59))

X = pima.iloc[:,features] # Features
y = pima.application_status_hired;
X = scale.fit_transform(X);
    
# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 ,  stratify=y, random_state=16)

# instantiate the model (using the default parameters)
#logreg = LogisticRegression(random_state=16)
logreg_default = LogisticRegression(solver='lbfgs', max_iter=1000)

# fit the model with data
logreg_default.fit(X_train, y_train)
y_pred = logreg_default.predict(X_test)


#bootsrap different models 

for i in range(n_iterations):
    bootstrapped_data =  pima.sample(frac=1, replace=True)
    
    #split dataset in features and target variable
    #features = [1] + list(range(4,15)) + list(range(16,20)) + list(range(23,59))
    #features = [1] + list(range(13,15)) + list(range(23,48))+ list(range(57,59)) +  list(range(62,71))
    
    X = bootstrapped_data.iloc[:,features] # Features
    y = bootstrapped_data.application_status_hired;
    X = scale.fit_transform(X);
        
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 ,  stratify=y, random_state=16)
    
    # instantiate the model (using the default parameters)
    #logreg = LogisticRegression(random_state=16)
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    # fit the model with data
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    logreg_array.append(logreg)
    y_pred_array.append(y_pred)
    y_test_array.append(y_test)
    
"""here i implement a confusion matrix to visualize the results, 
they are not that great but I'll focus on the next steps'
"""

  
cnf_matrix = metrics.confusion_matrix(y_test_array[1], y_pred_array[1])
cnf_matrix
    
    


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


"""now when it comes to predicting - lets do the following thought experiment: 
    each company is looking at a random 100 applicants and ranking them - 
    only the top 5 are considered. reducing systemacity could be seen as adding 
    more variety of applicants to this top 5 list 
    """
    
#create list of chosen ids/scores based on default/bootstrap
top5_ids_dlist = []
top5_scores_dlist = []  

top5_ids_blist = []
top5_scores_blist = []  
top5_scores_blist_adjusted = []  
    

n_hirings = 100;    
for i in range(n_hirings):
    #draw random applicant pool
        applicants =  pima.sample(n=10000, replace=False)
        
        candidate_ids = applicants.candidate_id 
        X = applicants.iloc[:,features] # Features
        y = applicants.application_status_hired;

        #get top 5 using default model
        hiring_probs = logreg_default.predict_proba(X)[:,1]
        top5 = np.argsort(hiring_probs)[::-1][:5]
        top5_ids = candidate_ids.iloc[top5]
        top5_scores = hiring_probs[top5]
        
        top5_ids_dlist.append(top5_ids)
        top5_scores_dlist.append(top5_scores)
        
        #get top 5 using bootstrapped models
        selected_logreg = random.choice(logreg_array)
        hiring_probs = selected_logreg.predict_proba(X)[:,1]
        
        top5 = np.argsort(hiring_probs)[::-1][:5]
        top5_ids = candidate_ids.iloc[top5]
        top5_scores = hiring_probs[top5]
        
        #I want to get the default model score for those candidates to compare
        hiring_probs_adjusted = logreg_default.predict_proba(X)[top5,1]
        top5_scores_adjusted = hiring_probs_adjusted; 
        
                
        top5_ids_blist.append(top5_ids)
        top5_scores_blist.append(top5_scores)
        top5_scores_blist_adjusted.append(top5_scores_adjusted)
        
        
        
import itertools
merged_d = list(itertools.chain(*top5_ids_dlist))        
merged_d_scores = list(itertools.chain(*top5_scores_dlist)) 
# converting our list to set
new_set = set(merged_d)
print("No of unique items in default model are:", len(new_set))

merged_b = list(itertools.chain(*top5_ids_blist))   
merged_b_scores = list(itertools.chain(*top5_scores_blist)) 
merged_b_scores_adjusted = list(itertools.chain(*top5_scores_blist_adjusted)) 

     
# converting our list to set
new_set = set(merged_b)
print("No of unique items in bootstrapped models are:", len(new_set))


import matplotlib.pyplot as plt


count_d = pd.Series(merged_d).value_counts()
count_d.reset_index(drop=True, inplace = True)
#print("Element Count:default ")
#print(count_d)
#count_d.plot(kind ='bar')


count_b = pd.Series(merged_b).value_counts()
count_b.reset_index(drop=True, inplace = True)
#print("Element Count:bootstapped")
#print(count_b)
#count_b.plot(kind ='bar')


'''
this graph shows number of 'hits' from our model for each applicant (sorted)
'''
fig, axes = plt.subplots(nrows=1,ncols=2)
count_d.plot(kind ='bar', ax = axes[0], title='default')
xmin, xmax = axes[0].get_xlim()
axes[0].set_xticks(np.round(np.linspace(int(xmin), int(xmax), 4), 2))
count_b.plot(kind ='bar', ax = axes[1], title='bootstrap')
xmin, xmax = axes[1].get_xlim()
axes[1].set_xticks(np.round(np.linspace(int(xmin), int(xmax), 4), 2))
plt.show()


#so we show the number of accepted applicants is higher, but is this coming 
#at expense of worse scores/probs?
import seaborn as sns

ax = sns.kdeplot(merged_d_scores)
sns.kdeplot(merged_b_scores, ax=ax)
plt.legend(labels=['default', 'bootstrapped'], title="scores")
plt.show()


ax = sns.histplot(merged_d_scores,  bins=len(merged_d_scores), stat="density",
             element="step", fill=False, cumulative=True, common_norm=False);
sns.histplot(merged_b_scores,  bins=len(merged_b_scores), stat="density",
             element="step", fill=False, cumulative=True, common_norm=False, ax=ax);
plt.legend(labels=['default', 'bootstrapped'], title="scores")
plt.show()

##the scores actually seem at times higher for the bootstrapped! however those are scores 
#from different models so they are not comparable, lets score them both on default model

ax = sns.kdeplot(merged_d_scores)
sns.kdeplot(merged_b_scores_adjusted, ax=ax)
plt.legend(labels=['default', 'bootstrapped'], title="scores")
plt.show()


ax = sns.histplot(merged_d_scores,  bins=len(merged_d_scores), stat="density",
             element="step", fill=False, cumulative=True, common_norm=False);
sns.histplot(merged_b_scores_adjusted,  bins=len(merged_b_scores), stat="density",
             element="step", fill=False, cumulative=True, common_norm=False, ax=ax);
plt.legend(labels=['default', 'bootstrapped'], title="scores")
plt.show()

'''
this was more in line with what we expect, the scores are lower for the bootstrapped
decision algorithm BUT it might not be the trade-off might not be so bad, we get more 
applicants - need to think of metrics for trade-off'
'''


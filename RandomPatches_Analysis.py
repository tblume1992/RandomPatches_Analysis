#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:11:16 2018

@author: tyler
"""

from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import pandas as pd
import numpy as np
from random import randint
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import mixture
import seaborn as sns





def RandomPatches_Analysis(dataset, y_variable, min_num_variables, iterations = 10000,
                           bootstrap_percentage = .7,  y_contain_0 = True):
    if y_contain_0:
        dataset[y_variable] += 1
    dataset[y_variable] = np.log(dataset[y_variable]) 
    dataset = dataset.dropna()
    train = dataset
    dataset = dataset.drop(y_variable, 1)
    
    
    #Build dataset which we will add the coefficients to
    coef_index_vals = list(dataset)
    coef_index_vals = pd.DataFrame(coef_index_vals).set_index([0], drop = True)
    count = 1
    coefficients = pd.DataFrame(index = coef_index_vals.index)
    #Random patches loop with progress bar
    mini_batch_count = 1
    mini_batch_coef = pd.DataFrame(index = coef_index_vals.index)
    
    for i in tqdm(range(iterations)):
        bootstrapped = train.sample(frac=bootstrap_percentage, replace=True, weights=None, random_state=None, axis=None)
        rand = randint(min_num_variables, len(coef_index_vals))
        y = bootstrapped[y_variable]
        X = bootstrapped.drop(y_variable,1)
        random_subspaces_X = X.sample(n=rand, replace=False, weights=None, random_state=None, axis=1)
        regent  = linear_model.LinearRegression()
        index_vals = list(random_subspaces_X)
        index_vals = pd.DataFrame(index_vals).set_index([0],drop = True)
        regent.fit(random_subspaces_X,y.values.ravel())
        coef = regent.coef_
        coef = pd.DataFrame(coef)
        coef.columns = [count]
        coef.index = index_vals.index
        mini_batch_coef = mini_batch_coef.join(coef, how='left', lsuffix='_left')
        if mini_batch_count == 200:
            coefficients = coefficients.join(mini_batch_coef, how='left', lsuffix='_left')
            mini_batch_coef = pd.DataFrame(index = coef_index_vals.index)
            mini_batch_count = 1
        count = count + 1
        mini_batch_count += 1
    coefficients = coefficients.join(mini_batch_coef, how='left', lsuffix='_left') 
    transposed_dataset = coefficients.T
    df = transposed_dataset.melt(var_name='Variables', value_name='Coefficients')
    
    

    
    
    variable_agg = []
    variable_means = []
    for variable in range(len(transposed_dataset.columns)):
        variable_ofinterest = transposed_dataset.iloc[:,variable].dropna()
        cluster_count = 1
        cluster_criterion = pd.DataFrame([])
        #Clustering to find the optimal number of distributions based on the BIC
        #Max number of clusters we can idnetify is 15 although it is unlikely it will ever be that high unless you have high dimensions
        #If it does look incorrect then just increases the number of random patches iterations
        while cluster_count < 15:
            clf = mixture.GaussianMixture(n_components=cluster_count, covariance_type='full')
            clf.fit(variable_ofinterest.values.reshape(-1, 1))
            bics = clf.bic(variable_ofinterest.values.reshape(-1, 1))
            cluster_criterion = cluster_criterion.append([bics],[cluster_count])
            cluster_count += 1
        
        #Re-cluster with the optimal number of clusters
        cluster_criterion = cluster_criterion.idxmin()
        number_of_clusters = cluster_criterion.at[0] + 1
        clf = mixture.GaussianMixture(n_components=number_of_clusters, covariance_type='full')
        clf.fit(variable_ofinterest.values.reshape(-1, 1))
        predictions = clf.predict(variable_ofinterest.values.reshape(-1, 1))
        variable_split = pd.DataFrame(np.column_stack([variable_ofinterest,predictions]))
        variable_split.columns = ['Coefficient', 'Cluster']
        #Get Means of the clusters
        distribution_means = variable_split.groupby(['Cluster'])[['Coefficient']].mean()
        distribution_means['Variable'] = np.tile(pd.DataFrame(transposed_dataset.iloc[:, variable]).columns, len(distribution_means))
        variable_means.append(distribution_means)
        variable_agg.append(variable_split)
        
    categorical_means = pd.concat(variable_means).round(decimals = 3)        

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)      
    sns.violinplot(x="Coefficients", y="Variables", data=df, ax = ax)
    sns.categorical.catplot(x="Coefficient", y="Variable", jitter = False, 
                            data=categorical_means, color = 'black', marker = 'v', s = 12, 
                            ax = ax)
    
    coefficients = coefficients.T
    variable_dataset = []
    for i in range(len(coefficients.columns)):
        coefficients_loop = coefficients.loc[coefficients.iloc[:,  i] > -1000]
        coefficients_ = coefficients_loop.drop(pd.DataFrame(coefficients_loop.iloc[:, i]).columns, axis = 1).T
        coefficient = coefficients_.fillna(0)
        coefficient = np.where(coefficient != 0, 1, 0)
        coefficient = pd.DataFrame(coefficient)
        variable_names = coefficients_.T
        variable_names = list(variable_names.columns.values)
        coefficient = coefficient.T
        coefficient_y = variable_agg[i]
        coefficient_y = coefficient_y['Cluster']
        columns = list(coefficients_.T)
        coefficient.columns = columns
        coefficient = [coefficient_y, coefficient]
        coefficient = np.column_stack(coefficient)
        coefficient = pd.DataFrame(coefficient)
        coefficient = coefficient.dropna()
        coefficient_y = coefficient[0]
        coefficient = coefficient.drop([0], 1)
        coefficient.columns = columns
        regent  = GaussianNB()
        regent.fit(coefficient,coefficient_y.values.ravel())
        variable_probabilities = regent.theta_
        variable_probabilities = pd.DataFrame(variable_probabilities)
        variable_probabilities.columns = variable_names
        variable_dataset.append(variable_probabilities)
    final_probs = pd.concat(variable_dataset) 
    final_probs[final_probs < 0.1] = 0
    final_probs[final_probs > 0.9] = 1
    final_probs = final_probs[final_probs.isin([0, 1])]
    final_probs[final_probs == 1] = 'Included'
    final_probs[final_probs == 0] = 'Dropped'
    final_probs = final_probs.fillna('-')


    final_probs['Expected Coefficient'] = categorical_means.Variable.str.cat(" ==> " +
               categorical_means.Coefficient.astype(str))
    print('Target Variable: ' + y_variable)
    print(final_probs.to_string(index=False))
    
    





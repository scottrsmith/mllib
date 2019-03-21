#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    ML Lib Producivity Class Library
    Copyright (C) 2019  Scott R Smith

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import print_function 
from __future__ import division

import mlLib.utility as utility

import pickle
import numpy as np
from time import time

# Import Elastic Net, Ridge Regression, and Lasso Regression
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression

# Import Random Forest and Gradient Boosted Trees
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#  classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# Function for creating model pipelines
from sklearn.pipeline import make_pipeline

# For standardization
from sklearn.preprocessing import StandardScaler

# Helper for cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError

# Import r2_score and mean_absolute_error functions
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, make_scorer, fbeta_score, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score

from sklearn.cluster import KMeans

# Classification metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.metrics import adjusted_rand_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.exceptions import DataConversionWarning
import warnings


#pd.options.mode.chained_assignment = None  # default='warn'

TRAIN_CLASSIFICATION = 'classification'
TRAIN_REGRESSION = 'regression'
TRAIN_CLUSTERING = 'clustering'


availableModels = { TRAIN_REGRESSION : ['lasso', 'ridge', 'enet', 'rf', 'gb', 'dtr', 'linearregression'],
                    TRAIN_CLASSIFICATION: ['l1', 'l2', 'rfc', 'gbc', 'decisiontree', 'kneighbors', 'sgd', 'bagging', 'adaboost', 
                                            'baggingbase', 'adaboostbase', 'gaussiannb'],
                    TRAIN_CLUSTERING: ['kmeans']}



def getModelPreferences(name, project, base=None):

    if name in project.overrideHyperparameters:
        override = project.overrideHyperparameters[name]
    else:
        override = None
        
        
    
    # Regression models
    if project.modelType == TRAIN_REGRESSION:
        if name=='lasso':
            return lassoPreferences(project,override)
        elif name=='ridge': 
            return ridgePreferences(project,override)
        elif name=='enet':
            return enetPreferences(project,override)
        elif name=='rf':
            return rfPreferences(project,override)
        elif name=='gb':
            return gbPreferences(project,override)
        elif name=='dtr':
            return dtrPreferences(project,override)
        elif name=='linearregression':
            return linearRegressionPreferences(project, override)
        
    
    # Now classification models
    if project.modelType == TRAIN_CLASSIFICATION:
        if name=='l1':
            return l1Preferences(project,override)
        elif name=='l2':
            return l2Preferences(project,override)
        elif name=='rfc':
            return rfcPreferences(project,override)
        elif name=='gbc':
            return gbcPreferences(project,override)
        elif name=='decisiontree':
            return dtcPreferences(project,override)
        elif name=='kneighbors':
            return kneighborsPreferences(project,override)
        elif name=='sgd':
            return sgdPreferences(project,override)
        elif name=='bagging':
            return baggingPreferences(project,override)
        elif name=='adaboost':
            return adaboostPreferences(project,override)
        elif name=='baggingbase':
            return baggingBasePreferences(project, override, base)
        elif name=='adaboostbase':
            return adaboostBasePreferences(project, override, base)
        elif name=='gaussiannb':
            return gaussiannbPreferences(project,override)        
        
    # and now clustering
    if project.modelType == TRAIN_CLUSTERING:
        if name=='kmeans':
            return kmeansPreferences(project,override)
    
    return None
    

#***************************
# regression
#***************************
  

# Lasso regressor
def lassoPreferences(project,override=None):
    # Lasso hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters == True:
        hyperparameters = {}
    elif project.hyperparametersLongRun == False: # Shorter run
        hyperparameters = { 
            'lasso__alpha' : [0.001, 0.01, 0.1, 1, 5, 10] 
            }
    else: # Longer run
        hyperparameters = { 
            'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
            }
    
    scorers = ['best','r2','MAE','score']
  
    return Lasso(random_state=project.randomState), hyperparameters, scorers

# Ridge Regressor
def ridgePreferences(project,override=None):
    # Ridge hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters == True:
        hyperparameters = {}
    elif project.hyperparametersLongRun == False: # Shorter run
        hyperparameters = { 
            'ridge__alpha': [0.001, 0.01, 0.1, 1, 5, 10]  
        }
    else: # Longer run
        hyperparameters = { 
            'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  
        }
     
    scorers = ['best','r2','MAE','score']
 
    return Ridge(random_state=project.randomState), hyperparameters, scorers

#Elastic Net Regression
def enetPreferences(project,override=None):
    # Elastic Net hyperparameters
    # Ridge hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters == True:
        hyperparameters = {}
    elif project.hyperparametersLongRun == False: # Shorter run
        hyperparameters = { 
            'elasticnet__alpha': [0.001, 0.01, 0.1, 1, 5, 10],                        
            'elasticnet__l1_ratio' : [0.1, 0.5, 0.9]  
        }
    else: # Longer run
        hyperparameters = { 
            'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],                        
            'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  
        }
    
    scorers = ['best','r2','MAE','score']
  
    return ElasticNet(random_state=project.randomState), hyperparameters, scorers

# Random forests regressor
def rfPreferences(project,override=None):
    # Random forest hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters == True:
        hyperparameters = {}
    elif project.hyperparametersLongRun == False: # Shorter run
        hyperparameters = {'randomforestregressor__n_estimators': [100, 200],
                            'randomforestregressor__max_features': ['auto', 'sqrt', 0.33]}
    else: # Longer run
        hyperparameters = {'randomforestregressor__n_estimators': [50, 100, 200, 500],
                            'randomforestregressor__max_features': ['auto', 'sqrt', 0.33]}
    
    scorers = ['best','r2','MAE','score']
 
    return RandomForestRegressor(random_state=project.randomState), hyperparameters, scorers

#hyperparameter grid for the boosted tree Regressor
def gbPreferences(project,override=None):
    # Boosted tree hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters == True:
        hyperparameters = {}
    elif project.hyperparametersLongRun == False: # Shorter run
        hyperparameters = {'gradientboostingregressor__n_estimators': [100, 200],
                           'gradientboostingregressor__learning_rate': [0.001, 0.01, 0.1],
                           'gradientboostingregressor__max_depth': [1, 5]}
    else: # Longer run
        hyperparameters = {'gradientboostingregressor__n_estimators': [50, 100, 200, 500],
                           'gradientboostingregressor__learning_rate': [0.001, 0.05, 0.1, 0.5],
                           'gradientboostingregressor__max_depth': [1, 5, 10, 50]}
    
    scorers = ['best','r2','MAE','score']
    return GradientBoostingRegressor(random_state=project.randomState), hyperparameters, scorers

#hyperparameter grid for the boosted tree Regressor
def dtrPreferences(project,override=None):
    # Boosted tree hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters == True:
        hyperparameters = {}
    elif project.hyperparametersLongRun == False: # Shorter run
        hyperparameters = {'decisiontreeregressor__max_depth':[1, 8, 32]}
    else: # Longer run
        hyperparameters = {'decisiontreeregressor__max_depth':[1, 8, 16, 32, 64, 200]}
    
    scorers = ['best','r2','MAE','score']
    return DecisionTreeRegressor(random_state=project.randomState), hyperparameters, scorers

#hyperparameter grid for the boosted tree Regressor
def linearRegressionPreferences(project, override=None):
    # Boosted tree hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters == True:
        hyperparameters = {}
    elif project.hyperparametersLongRun == False: # Shorter run
        hyperparameters = {}
    else: # Longer run
        hyperparameters = {}
    
    
    scorers = ['best','r2','MAE','score']
    
    return LinearRegression(), hyperparameters, scorers

   


#***************************
# Classifiers
#***************************
def l1Preferences(project,override=None):
    # for  L1L1 -regularized logistic regression
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # lgon run
        hyperparameters = {
                'logisticregression__solver' : ['liblinear', 'saga'],
                'logisticregression__C' : [.0001,.001, .01, .1, 10, 30, 50 ,100, 250, 500, 1000],
                'logisticregression__max_iter': [15, 25, 50, 100, 300, 500]
                }
    else: # short run
        hyperparameters = {
                'logisticregression__solver' : ['liblinear'],
                'logisticregression__max_iter': [10, 15, 25], 
                'logisticregression__C' : [.01, .1, 10, 50]
                }
    
    if project.modelType==TRAIN_REGRESSION:
        scorers = ['best','r2','MAE','accuracy','score']
    elif project.modelType==TRAIN_CLASSIFICATION:
        scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    else:
        scorers = []
        
    return LogisticRegression(penalty='l1' ,random_state=project.randomState), hyperparameters, scorers



def l2Preferences(project,override=None):
    # for  L2L2 -regularized logistic regression
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # long run
        hyperparameters = {
                'logisticregression__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'logisticregression__C' : [.0001,.001, .01, .1, 10, 50, 100, 250, 500, 1000],
                'logisticregression__max_iter': [15, 25, 50, 100, 300, 500]
                }
    else: # short run
        hyperparameters = {
                'logisticregression__solver' : ['lbfgs', 'liblinear','sag'],
                'logisticregression__max_iter': [20, 25, 30], 
                'logisticregression__C' : [.001, .01, .1, 1.0, 10.]
                }
    
    if project.modelType==TRAIN_REGRESSION:
        scorers = ['best','r2','MAE','accuracy','score']
    elif project.modelType==TRAIN_CLASSIFICATION:
        scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    else:
        scorers = []
        
    return LogisticRegression(penalty='l2' ,random_state=project.randomState), hyperparameters, scorers



def rfcPreferences(project,override=None):
    # Random forest hyperparameters
    
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # long run
        hyperparameters = {
                'randomforestclassifier__n_estimators': [10, 50, 100, 200],
                'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33, .11, 1.0]
                           }
    else: # short run
        hyperparameters = {'randomforestclassifier__n_estimators': [200, 250],
                'randomforestclassifier__max_features': [1.0, .80, 0.33]}
    
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return RandomForestClassifier(random_state=project.randomState), hyperparameters, scorers



# hyperparameter grid for the  gradient boosted tree Classifier
def gbcPreferences(project, override=None):
    #  gradient boosted tree
    
    if override is not None:
        hyperparameters = override
    elif override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # long run
        hyperparameters = {
                'gradientboostingclassifier__min_samples_split': [2, 100, 500],
                'gradientboostingclassifier__min_samples_leaf': [1, 50],
                'gradientboostingclassifier__max_features': ['sqrt'],
                'gradientboostingclassifier__subsample': [1., .8],
                'gradientboostingclassifier__n_estimators': [175, 225],
                'gradientboostingclassifier__max_depth': [6, 8, 10, 12],
                'gradientboostingclassifier__loss': ['exponential','deviance'],
                'gradientboostingclassifier__learning_rate':[.2, .1, .01, .05],
                'gradientboostingclassifier__validation_fraction' :[0.1],
                'gradientboostingclassifier__n_iter_no_change' :[5,10],
                'gradientboostingclassifier__tol':[.1, .01, 0.001]
                }
    else: # short run
        hyperparameters = {
                'gradientboostingclassifier__n_estimators': [300, 500],
                'gradientboostingclassifier__min_samples_split': [500],
                'gradientboostingclassifier__loss': ['deviance'],
                'gradientboostingclassifier__min_samples_leaf': [25],
                'gradientboostingclassifier__max_features': ['sqrt'],
                'gradientboostingclassifier__max_depth': [6, 8],
                'gradientboostingclassifier__learning_rate':[1., .1, .01],
                'gradientboostingclassifier__subsample': [.8],
                'gradientboostingclassifier__validation_fraction' :[0.1],
                'gradientboostingclassifier__n_iter_no_change' :[10],
                'gradientboostingclassifier__tol':[0.001]
        }
    
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return GradientBoostingClassifier(random_state=project.randomState), hyperparameters, scorers

#hyperparameter grid for the boosted tree 
def dtcPreferences(project, override=None):
    # Boosted tree hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # Shorter run
        hyperparameters = {
                'decisiontreeclassifier__criterion' :['gini', 'entropy'],
                'decisiontreeclassifier__splitter' :['best', 'random'],
                'decisiontreeclassifier__min_samples_split' :[2, 50, 100],
                'decisiontreeclassifier__min_samples_leaf': [1, 50, 100],
                'decisiontreeclassifier__max_features': ['sqrt','auto','log2',1., .2, .5],
                'decisiontreeclassifier__max_depth':[None, 1, 10, 50, 100, 250]
                }
    else: # Longer run
        hyperparameters = {'decisiontreeclassifier__splitter': ['best'], 
                                     'decisiontreeclassifier__min_samples_leaf': [100, 150], 
                                     'decisiontreeclassifier__min_samples_split': [100, 150], 
                                     'decisiontreeclassifier__criterion': ['entropy'], 
                                     'decisiontreeclassifier__max_features': [1.0], 
                                     'decisiontreeclassifier__max_depth': [10]}
    
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return DecisionTreeClassifier(random_state=project.randomState), hyperparameters, scorers


#hyperparameter grid  
def kneighborsPreferences(project,override=None):
    # hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # Shorter run
        hyperparameters = {}
    else: # Longer run
        hyperparameters = {}
    
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return KNeighborsClassifier(), hyperparameters, scorers

#hyperparameter grid  
def sgdPreferences(project,override=None):
    # hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
                             # Loss: 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'
                            # regression loss: 'squared_loss', 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
    elif project.hyperparametersLongRun: # long run
        hyperparameters = {
                'sgdclassifier__loss': ['hinge', 'squared_loss', 'perceptron'], # 
                'sgdclassifier__penalty': ['l2', 'elasticnet'], # 'none', 'l2', 'l1', or 'elasticnet'
                'sgdclassifier__max_iter': [10, 100, 1000],
                'sgdclassifier__tol': [1e-1 , 1e-2 , 1e-3 ] 
                }
                        
    else: # short run
        hyperparameters = {
                'sgdclassifier__loss': ['hinge'], # 
                'sgdclassifier__penalty': ['l2'], # 'none', 'l2', 'l1', or 'elasticnet'
                'sgdclassifier__max_iter': [1000],
                'sgdclassifier__tol': [1e-3 ]
                }
    
    scorers = ['best','r2','MAE','accuracy','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return SGDClassifier(random_state=project.randomState), hyperparameters, scorers

#hyperparameter grid  
def baggingPreferences(project,override=None):
    # hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # long run
        hyperparameters = {
                'baggingclassifier__n_estimators':[5, 10, 100], 
                'baggingclassifier__max_samples':[.5, .75, .95, 1.], 
                'baggingclassifier__max_features':[ .50, .75, .95, 1.]
                }
    else: # short run
        hyperparameters = {
               'baggingclassifier__max_features': [.75, 1.], 
               'baggingclassifier__max_samples': [.75, 1.], 
               'baggingclassifier__n_estimators': [50, 100]}
     
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return BaggingClassifier(random_state=project.randomState), hyperparameters, scorers


#hyperparameter grid  
def adaboostPreferences(project,override=None):
    # hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # long run
        hyperparameters = {
                'adaboostclassifier__n_estimators':[10, 50, 100], 
                'adaboostclassifier__learning_rate':[.1, 0.5, 1.0], 
                'adaboostclassifier__algorithm':['SAMME', 'SAMME.R']
                }
    else: # short run
        hyperparameters = {'adaboostclassifier__n_estimators':[10, 100], 
                'adaboostclassifier__learning_rate':[.9, 1.0], 
                'adaboostclassifier__algorithm':['SAMME.R']}
     
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return AdaBoostClassifier(random_state=project.randomState), hyperparameters, scorers



#hyperparameter grid  
def baggingBasePreferences(project, override=None, baseEstimator=None):
    # hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # longf run
        hyperparameters = {}
    else: # short run
        hyperparameters = {}   
        
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    
    if baseEstimator is not None:
        base = baseEstimator
    else:
        base = LogisticRegression(penalty='l2' , C=5, max_iter=15, solver='newton-cg',random_state=project.randomState)
        
    return BaggingClassifier(base_estimator=base,random_state=project.randomState), hyperparameters, scorers



#hyperparameter grid  
def adaboostBasePreferences(project, override=None, baseEstimator=None):
    # hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # Shorter run
        hyperparameters = {}
    else: # Longer run
        hyperparameters = {}
    
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    
    if baseEstimator is not None:
        base = baseEstimator
    else:
        base = LogisticRegression(penalty='l2' , C=5, max_iter=15, solver='newton-cg',random_state=project.randomState)
        
    return AdaBoostClassifier(base_estimator=base,random_state=project.randomState), hyperparameters, scorers


#hyperparameter grid  
def gaussiannbPreferences(project,override=None):
    # hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: # long run
        hyperparameters = {'gaussiannb__var_smoothing':[ 1., 1e-2, 1e-1, 1e-3, 1e-5, 1e-9]}
    else: # short run
        hyperparameters = {'gaussiannb__var_smoothing':[ 1.0, .01, .001]}
    
    scorers = ['best','r2','MAE','accuracy','auroc','fbeta','score','confusionmatrix', 'f1', 'recall', 'precision']
    return GaussianNB(), hyperparameters, scorers



#***************************
# clustering
#***************************


#hyperparameter grid for the boosted tree Regressor
def kmeansPreferences(project, override=None):
    # Boosted tree hyperparameters
    if override is not None:
        hyperparameters = override
    elif project.defaultHyperparameters:
        hyperparameters = {}
    elif project.hyperparametersLongRun: #
        hyperparameters = {}
    else: # short run
        hyperparameters = {}

    scorers = []
    return KMeans(n_clusters=project.kmeansClusters, random_state=project.randomState), hyperparameters, scorers

 

class trainModels (object):
    
    def __init__ (self, tableName, project):
    
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)


        # Split X and y into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = project.preppedData[tableName].getTrainingSet()

        #print (len(self.X_train), len(self.X_test), len(self.y_train), len(self.y_test) )
    
        #print (self.X_train.describe())
    
    
        self.project = project
        self.pipelines = {}
        self.hyperparameters = {}
        self.scoringList = {}
        self.modelListAsRun = None
        project.modelListAsRun = []
        
        for name in project.modelList:
            if (name=='baggingbase' or name=='adaboostbase') and project.baseEstimator is not None:
                for baseName, base in project.baseEstimator:
                    newName = name+'+'+baseName
                    project.modelListAsRun.append(newName)
                    #utility.runLog('Prepping: {} with {} '.format(name,base))
                    model, hyperparams, modelScoringList = getModelPreferences(name, project, base)
                    if project.useStandardScaler:
                        self.pipelines[newName] = make_pipeline(StandardScaler(), model)
                    else:
                        self.pipelines[newName] = make_pipeline(model)
                    self.hyperparameters[newName] = hyperparams
                    self.scoringList[newName] = modelScoringList
                    
            
            else:
                project.modelListAsRun.append(name)
                model, hyperparams, modelScoringList = getModelPreferences(name, project)
                if project.useStandardScaler:
                    self.pipelines[name] = make_pipeline(StandardScaler(), model)
                else:
                    self.pipelines[name] = make_pipeline(model)
                self.hyperparameters[name] = hyperparams
                self.scoringList[name] = modelScoringList
            

        self.modelType = project.modelType
        self.modelList = project.modelList
        self.modelListAsRun = project.modelListAsRun
        self.fittedModels = {}
        self.modelScores = {}
        self.shortModelScoresColumns = []
        
        self.bestModelName = None
        self.bestModelScore = None
        self.bestModel = None
        self.crossValidationSplits = project.crossValidationSplits
        self.parallelJobs = project.parallelJobs
        self.tableName = tableName


        
    def fitModels(self):

        warnings.filterwarnings(action='ignore', category=DataConversionWarning)


        # Create empty dictionary called fitted_models
        self.fittedModels = {}
        self.runTimes = {}
               
        # Loop through model pipelines, tuning each one and saving it to fitted_models
        for name, pipeline in self.pipelines.items():
            # Create cross-validation object from pipeline and hyperparameters
            runTimeStart = time()
            utility.runLog('\n\nFitting: '+name)

            if self.modelType==TRAIN_CLUSTERING:
                model = pipeline
                model.fit(self.X_train)
            else:
                model = GridSearchCV(pipeline, self.hyperparameters[name], cv=self.crossValidationSplits, 
                                    n_jobs=self.parallelJobs, verbose=self.project.gridSearchVerbose,
                                    scoring=self.project.gridSearchScoring)
                model.fit(self.X_train, self.y_train)
 
        
            # Fit model on X_train, y_train
#            print ()
#            print (('--------------------->', name))
#            print (pipeline.get_params())
#            print (self.X_train.describe())
            
            
            # Store model in fitted_models[name] 
            self.fittedModels[name] = model
            self.runTimes[name] = (time() - runTimeStart) / 60.
            
            self.scoreModels(name, model)    
                    
            if self.modelType!=TRAIN_CLUSTERING:
                utility.runLog("\n\n{} has been fit. The 'best' score is {:.4f} with a runtime of {:.3f} Minutes".format(
                        name, model.best_score_,self.runTimes[name]))
                utility.runLog('   Best Paramaters= {}'.format(model.best_params_))
                self.project.logTrainingResults(self.tableName, 'runLog.csv', name)
            
       
            # Print out the current accumulated scores
            if self.project.ongoingReporting:
                self.project.displayAllScores(self.project.ongoingReportingFilename, short=True)       
         
        self.getBestModel(self.project)
        return
        
        
        
    def scoreModels(self, name, model):
            
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
        
        confusionMatrix = None
        fpr, tpr, thresholds = None, None, None
        
        r2 = None
        MAE = None
        auroc = None
        rocauc_score = None
        best = None    
        accuracy = None
        fbeta = None
        score = None
            
        predProba = None
        pred = None
            
        f1, recall, precision = None, None, None
            
            
        try:
                                       
            # Get each score based upon the scorelist initilized for the model
            for scorer in self.scoringList[name]:
                if scorer == 'best':
                    best = model.best_score_
                        
                elif scorer == 'r2':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    r2 = r2_score(self.y_test, pred )
                        
                elif scorer == 'MAE':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    MAE = mean_absolute_error(self.y_test, pred)
                    
                elif scorer == 'f1':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    f1 = f1_score(self.y_test, pred)

                elif scorer == 'recall':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    recall = recall_score(self.y_test, pred)

                elif scorer == 'precision':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    precision = precision_score(self.y_test, pred)

                elif scorer == 'accuracy':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    accuracy = accuracy_score(self.y_test, pred)
                        
                elif scorer == 'auroc':
                    if predProba is None:
                        predProba = model.predict_proba(self.X_test)
                    # Area Under Receiver operating characteristic (AUROC)
                    rows, cols = predProba.shape
                    if cols==2:
                        aurocPred = [p[1] for p in predProba]
                    else:
                        aurocPred = predProba
                    fpr, tpr, thresholds = roc_curve(self.y_test, aurocPred)
                    auroc =  auc(fpr, tpr)
                    rocauc_score = roc_auc_score(self.y_test, aurocPred)
                        
                elif scorer == 'fbeta':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    fbeta = fbeta_score(self.y_test, pred, beta = self.project.fbeta)
                        
                elif scorer == 'score':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    score = model.score(self.X_test, pred)
                       
                elif scorer == 'confusionmatrix':
                    if pred is None:
                        pred = model.predict(self.X_test)
                    confusionMatrix = confusion_matrix(self.y_test, pred)
                        #confusionMatrix = confusionMatrix.astype('float')/confusionMatrix.sum(axis=0)
                                
                # Record the scores
            self.modelScores[name]= {'r2':r2, 'MAE':MAE, 'Best':best, 'AUROC': auroc, 'Accuracy': accuracy, 
                                    'rocauc_score': rocauc_score,'fbeta': fbeta, 'roc_curve':(fpr, tpr, thresholds),
                                    'CM':confusionMatrix, 'Score':score, 'F1':f1, 'Recall': recall,
                                    'Precision': precision, 'RunTime': self.runTimes[name]} 
                                    
            self.shortModelScoresColumns = ['r2','MAE','Best','AUROC','Accuracy','fbeta',
                                            'Score','F1', 'Recall', 'Precision', 'RunTime']
                                                                    
        except NotFittedError as e:
            print (repr(e))
        
        return
 


               
        # Display best_score_ for each fitted model
        #for name, model in fittedModels.items():
        #   print( name, model.best_score_ )


    def getBestModel(self, project):
        if self.modelType==TRAIN_CLASSIFICATION:
            theBest = self.getBestModelClassification()
            self.bestModel = self.fittedModels[theBest].best_estimator_
        elif self.modelType==TRAIN_REGRESSION:
            theBest = self.getBestModelRegression()
            self.bestModel = self.fittedModels[theBest].best_estimator_
        elif self.modelType==TRAIN_CLUSTERING:
            theBest = self.getBestModelClustering()
            self.bestModel = self.fittedModels[theBest]
        
        self.bestModelName = theBest
        self.bestModelScore = self.modelScores[theBest]
        
        project.bestModelName = self.bestModelName
        project.bestModelScore = self.bestModelScore
        project.bestModel = self.bestModel
 
        return 


    """
    Which model had the highest  R2R2  on the test set?
    Random forest
    
    Which model had the lowest mean absolute error?
    Random forest
    
    Are these two models the same one?
    Yes
    
    Did it also have the best holdout  R2R2  score from cross-validation?
    Yes
    
    Does it satisfy our win condition?
    Yes, its mean absolute error is less than $70,000!
    """

    def getBestModelRegression(self):
        # highest  R2R2  on the test set
        # lowest mean absolute error
        # Best holdout R2 (best score)
        highestR2 = None
        lowestMAE = None
        bestHoldoutR2 = None
        highestR2Name = None
        lowestMAEName = None
        bestHoldoutR2Name = None
        
        for name in self.modelScores:
            scores = self.modelScores[name]
            r2 = scores['r2']
            MAE = scores['MAE']
            Best = scores['Best']
            if highestR2 == None:
                highestR2 = r2
                lowestMAE = MAE
                bestHoldoutR2 = Best
                highestR2Name = name
                lowestMAEName = name
                bestHoldoutR2Name = name
            else:
                if r2 >  highestR2:
                    highestR2 = r2
                    highestR2Name = name
                if MAE < lowestMAE:
                    lowestMAE = MAE
                    lowestMAEName = name
                if Best > bestHoldoutR2:
                    bestHoldoutR2 = Best
                    bestHoldoutR2Name = name
        
        if ((highestR2Name == lowestMAEName) and (lowestMAEName == bestHoldoutR2Name)):
            theBest = highestR2Name
        elif (highestR2Name == lowestMAEName):
            theBest = highestR2Name
        elif (lowestMAEName == bestHoldoutR2Name):
            theBest = lowestMAEName
        elif (highestR2Name == bestHoldoutR2Name):
            theBest = highestR2Name
        else:
            theBest = bestHoldoutR2Name
                              
        return theBest

    def getBestModelClassification(self):
        # highest  R2R2  on the test set
        # lowest mean absolute error
        # Best holdout R2 (best score)
        highestAUROC = -np.inf
        bestAcc = -np.inf
        theBest = None
        lastName = None
        
        for name in self.modelScores:
            lastName = name
            scores = self.modelScores[name]
            auroc = scores['AUROC']
            acc = scores['Accuracy']
            if auroc is not None:
                if auroc >  highestAUROC:
                    highestAUROC = auroc
                    theBest = name
            #elif acc > bestAcc:
            #    bestAcc = acc
            #    theBest = name
                
        if theBest is None:
            theBest = lastName
        return theBest
 

    def getBestModelClustering(self):
       
        for name in self.modelScores:
            return name
    





#for name, model in fitted_models.items():
#    pred = model.predict(X_test)
#    print ( name )
#    print ( '--------' )
#    print ( 'R^2:', r2_score(y_test, pred ))
#    print ( 'MAE:', mean_absolute_error(y_test, pred))
#    print ()
#    
#gb_pred = fitted_models['rf'].predict(X_test)
#plt.scatter(gb_pred, y_test)
#plt.xlabel('predicted')
#plt.ylabel('actual')
#plt.show()

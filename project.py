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


import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path


from .getData import getData
from .exploreData import exploreData
from .cleanData import cleanData, cleaningRules
from .prepData import prepData, prepPredictData
import mlLib.trainModels as tm
import mlLib.utility as utility



import pickle as pk
from sklearn.exceptions import NotFittedError


"""

mlProject is the top level object for training and running a ML project.
Various object mothods are used to load, review, and train the data, as well as manage running predictions

Example:
project = mlProject('Customer Segements', 'clustering model should factor in both aggregate sales patterns and specific items purchased')


"""
class mlProject (object):
    
    def __init__ (self, name, description=None):
        
        self.name = name
        self.description = description
        self.dataFile = {}
        self.preppedTablesDF = {}
        self.defaultPreppedTableName = None
        self.batchTablesList = {}
        self.explore = {}
        self.cleaningRules = {}
        
        # Training variables
        self.testSize = .2
        self.randomState = 1234
        
        # training prep data - preppedData class
        self.preppedData = {}
        
        # Training variable for tables
        self.targetVariable = {}
        self.targetVariableIsBoolean = {}
        self.targetVariableTrueValue = {}
        self.targetVariableConvertValues = {}
 
    
        # Managing object values
        self.uniqueThreshold = 50
        self.smallSample = 25
        self.highDimensionality = 100
        
        # Clustering
        self.varianceThreshold = .8
        self.clusterDimensionThreshold = 20
        
        # kmeans defaults
        self.kmeansClusters = 3
        self.useStandardScaler = True
        
        
        self.dropDuplicates = True

        self.trainedModels = {}
        self.modelScores = None
        self.bestModelName = None
        self.bestModelScore = None
        self.bestModel = None
        
        
        # model data
        self.crossValidationSplits = 10
        self.parallelJobs = -1
        self.modelType = None
        self.modelList = None
        self.modelListAsRun = None
        
        self.defaultHyperparameters = None
        self.hyperparametersLongRun = None
        self.overrideHyperparameters={}
        self.baseEstimator=None
        
        # Reporting
        self.confusionMatrixLabels = None
        self.ongoingReporting = False
        self.ongoingReportingFilename = None
        
        # Gridsearch Variables
        self.gridSearchVerbose = 0
        self.gridSearchScoring = None
        
        # Goals
        self.goalsToReach = None
        return


    """
    Purpose: To set the training prerfferences for a project. This sets the type of training, regression, classification, clustering and
             the models used. There are also ways to set the hyperparamaters
        
    Call Variables:
    def setTrainingPreferences (self, crossValidationSplits=None, parallelJobs=None, modelType=None, modelList=None, 
                                testSize=None, randomState=None, uniqueThreshold=None, dropDuplicates=None, 
                                clusterDimensionThreshold=None, varianceThreshold=None, kmeansClusters=None,  useStandardScaler = None,
                                fbeta=None, defaultHyperparameters=None, hyperparametersLongRun=None, gridSearchVerbose=0,
                                gridSearchScoring=None):
    Example:
        project.setTrainingPreferences (crossValidationSplits=10, parallelJobs=-1, modelType=tm.TRAIN_CLASSIFICATION, 
            modelList=['l1', 'l2', 'rfc', 'gbc', 'kneighbors', 'sgd', 'bagging', 'adaboost', 'gaussiannb'] ) 
 
 
    Models to be used:
        TRAIN_REGRESSION : ['l1', 'l2', 'lasso', 'ridge', 'enet', 'rf', 'gb', 'decisiontree', 'linearregression'],
        TRAIN_CLASSIFICATION: ['l1', 'l2', 'rfc', 'gbc', 'decisiontree', 'kneighbors', 'sgd', 'bagging', 
                                'adaboost', 'gaussiannb', 'linearregression'],
        TRAIN_CLUSTERING: ['kmeans']}
     
    """    
    def setTrainingPreferences (self, crossValidationSplits=None, parallelJobs=None, modelType=None, modelList=None, 
                                testSize=None, randomState=None, uniqueThreshold=None, dropDuplicates=None, 
                                clusterDimensionThreshold=None, varianceThreshold=None, kmeansClusters=None,  useStandardScaler = None,
                                fbeta=None, defaultHyperparameters=None, hyperparametersLongRun=None, gridSearchVerbose=0,
                                gridSearchScoring=None):
                                
        if crossValidationSplits is not None:
            self.crossValidationSplits = crossValidationSplits
        if parallelJobs is not None:
            self.parallelJobs = parallelJobs
            
            
        if modelType is not None:
            if modelType in tm.availableModels:
                self.modelType = modelType
            else:
                utility.raiseError(modelType + ' is not a valid model type')
            
            
        if modelList is not None:
            for x in modelList:
                if x not in tm.availableModels[self.modelType]:
                    utility.raiseError('Model {} not found'.format(x))
            self.modelList = modelList
        elif self.modelType is not None:
            self.modelList = tm.availableModels[self.modelType]
            
            
        if useStandardScaler is not None:
            self.useStandardScaler = useStandardScaler
        
        if fbeta is not None:
            self.fbeta = fbeta
        else:
            self.fbeta = 1.0
            
        if defaultHyperparameters is not None:
            self.defaultHyperparameters = defaultHyperparameters
        else:
            self.defaultHyperparameters = False

        # The complexity of hyperparameters, thus the length of how long the run.
        # True = Longer run
        if hyperparametersLongRun is not None:
            self.hyperparametersLongRun = hyperparametersLongRun
        else:
            self.hyperparametersLongRun = False
  
            
        if testSize is not None:
            self.testSize = testSize
        if randomState is not None:
            self.randomState = randomState
        if uniqueThreshold is not None:
            self.uniqueThreshold = uniqueThreshold
        if dropDuplicates is not None:
            self.dropDuplicates = dropDuplicates
            
            
        if clusterDimensionThreshold is not None:
            self.clusterDimensionThreshold = clusterDimensionThreshold
        if varianceThreshold is not None:
            self.varianceThreshold = varianceThreshold
        if kmeansClusters is not None:
            self.kmeansClusters = kmeansClusters
            
        self.gridSearchVerbose = gridSearchVerbose
        
        # Set the scoring function 
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter       
        
        if gridSearchScoring is None:
            if self.modelType==tm.TRAIN_REGRESSION:
                self.gridSearchScoring = 'r2'
            elif self.modelType==tm.TRAIN_CLASSIFICATION:
                self.gridSearchScoring = 'f1'
            else: #tm.TRAIN_CLUSTERING
                self.gridSearchScoring = None
        else:
            self.gridSearchScoring = gridSearchScoring

    """
    Purpose:        
    Set the hyperparameters to override the defaults for a model
    
    Example:
            hyperparameters = { 
                'lasso__alpha' : [0.001, 0.01, 0.1, 1, 5, 10] 
                }
            project.setHyperparametersOverride(self, 'lasso', hyperparameters)
            
            
        hyperparameters = { 
            'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
            }
        hyperparameters = { 
            'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  
        }
        hyperparameters = { 
            'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],                        
            'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  
        }
        hyperparameters = {'randomforestregressor__n_estimators': [50, 100, 200, 500],
                            'randomforestregressor__max_features': ['auto', 'sqrt', 0.33]}
        hyperparameters = {'gradientboostingregressor__n_estimators': [50, 100, 200, 500],
                           'gradientboostingregressor__learning_rate': [0.001, 0.05, 0.1, 0.5],
                           'gradientboostingregressor__max_depth': [1, 5, 10, 50]}
        hyperparameters = {'decisiontreeregressor__max_depth':[1, 8, 16, 32, 64, 200]}
        hyperparameters = {
                'logisticregression__C' : np.linspace(1e-4, 1e3, num=50),
                'logisticregression__max_iter': [25, 50, 100, 300, 500]
                }
        hyperparameters = {
                'logisticregression__C' : np.linspace(1e-4, 1e3, num=50),
                'logisticregression__max_iter': [25, 100, 300, 500]
                }
        hyperparameters = {'randomforestclassifier__n_estimators': [100, 200],
                            'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33]}
        hyperparameters = {'gradientboostingclassifier__n_estimators': [50, 100, 200, 500],
            'gradientboostingclassifier__max_depth': [1, 10, 50, 100],
            'gradientboostingclassifier__learning_rate':[.1, .01, .001, .0001]}
            
    """
    def setHyperparametersOverride(self, modelName, override):
        self.overrideHyperparameters[modelName] = override
     
     
    """
        Purpose: Set the hyperpatamaters for the base_estimator 
        
        For: adaboost or bagging
        
        Example:
        
        from sklearn.linear_model import LogisticRegression
        project.setBaseHyperparametersOverride('bagging', 
                 LogisticRegression(penalty='l1', C=np.linspace(1e-4, 1e3, num=20),
                 random_state=project.randomState))
            
    """
    def setBaseEstimator(self, model):
        self.baseEstimator=model
  
    
    """

        Example: project.setConfusionMatrixLabels([(0,'Paid'), (1, 'Default') ])

        
    """
    def setConfusionMatrixLabels(self,list):
        self.confusionMatrixLabels = list
        return
    
    
 
    """
    Purpose: Set the target variable for supervised learning. 
            
    Call: setTarget(self, value, boolean=False, trueValue=None, convertTable=None, tableName=None):
            
    Example:
            project.setTarget('loan_status') 
    
        
        trueValue = what is the true values
        boolean = is this a boolean value
        convertTable = a table of how to convert values
            
    """
     
    def setTarget(self, value, boolean=False, trueValue=None, convertTable=None, tableName=None):
        if tableName is not None:
            if tableName in self.preppedTablesDF:
                theName = tableName
        else:
            theName = self.defaultPreppedTableName
        self.targetVariable[theName] = value
        self.targetVariableIsBoolean[theName] = boolean
        self.targetVariableTrueValue[theName] = trueValue
        self.targetVariableConvertValues[theName] = convertTable        
        return
    
    
    """
        def importFile(self, name, type=None, description=None, location=None, fileName=None, sheetName=None, hasHeaders = False, 
                      range=None, isDefault=False):
        
        
        project.importFile('Loan Data', type='csv', description='Lending Club Data from 2017-2018', 
                fileName='LendingClub2017_2018ready.csv',  hasHeaders = True, isDefault=True)
        
        
        
    """
    def importFile(self, name, type=None, description=None, location=None, fileName=None, sheetName=None, hasHeaders = False, range=None, isDefault=False):
        
        self.dataFile[name] = getData(name, type=type, description=description, location = location, fileName=fileName,  sheetName=sheetName, range=range, hasHeaders = hasHeaders)
        self.preppedTablesDF[name] = self.dataFile[name].openTable()
        if isDefault:
            self.defaultPreppedTableName = name
        elif self.defaultPreppedTableName is None:
            self.defaultPreppedTableName = name

    """
    Purpose: Export the named file. (Projects can have multiuple files associated with them)
            
    Call: def exportFile(self, name, filename):
            
    Example: project.exportFile('Loan Data', 'fileout.csv'):
            
    """
     def exportFile(self, name, filename):
        if name in self.preppedTablesDF:
            self.preppedTablesDF[name].to_csv(filename, index=False)
        return



    """
    Purpose: Run the explore data function. This will review the data and make recommendations
            
    Call: exploreData(self):
            
    Example: project.exploreData()
            
    """
    def exploreData(self):
        for name in self.preppedTablesDF:
           self.explore[name] = exploreData(self.preppedTablesDF[name], self)

           
    """
           Before adding any cleaning rules you must init
           
           project.initCleaningRules()


           project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET, 'term', [['36 months', ' 36 months'], '36'])
           project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET, 'term', [['60 months', ' 60 months'], '60'])
           
    """
    def initCleaningRules(self):
        for name in self.preppedTablesDF:
            self.cleaningRules[name] = cleaningRules(self, self.explore[name])


    # Just run the cleaning rules - do not explore


    """
    Purpose: Run the cleaning rules established for a project. 
            
    Call: cleanProject(self)
            
    Example: project.cleanProject()
            
    """
    def cleanProject(self):
        toClean = [x for x in self.preppedTablesDF]
        for name in toClean:
            cleanData(self.preppedTablesDF[name], self.cleaningRules[name])
        return 



    """
    Purpose: Run clean and explore together
            
    Call: def cleanAndExploreProject(self)
            
    Example: project.cleanAndExploreProject()
            
    """
     def cleanAndExploreProject(self):
        toClean = [x for x in self.preppedTablesDF]
        for name in toClean:
            cleanData(self.preppedTablesDF[name], self.cleaningRules[name])
             
        for name in self.preppedTablesDF:
           self.explore[name] = exploreData(self.preppedTablesDF[name], self)
        return 



    """
    Purpose: Prepare the 'table' for training. This will one-hot encode, for example
            
    Call: prepProjectByName(self, tableName=None)
            
    Example: project.prepProjectByName('Loan Data')
            
    """
     def prepProjectByName(self, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
 
        if theName in self.preppedTablesDF:
            self.preppedData[theName] = prepData(theName, self)
        return
        
        
        
    """
    Purpose: Once a file has been cleaned and explorred
            
    Call:
            
    Example:
            
    """
    def writePreppedFileByName(self, filename, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
        if theName in self.preppedTablesDF:
            utility.runLog('Exporting Prepped Data '+theName)
            self.preppedTablesDF[theName].to_csv(filename,index=False)
        return
        
        
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
    def writeTrainingSetFileByName(self, filename, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
        if theName in self.preppedData:
            X_train, X_test, y_train, y_test = self.preppedData[theName].getTrainingSet()
            utility.runLog('Exporting Training Set '+theName+' to file '+filename)
            X_train.to_csv(filename,index=False)
        return


    """
    Purpose:
            
    Call:
            
    Example:
            
    """
     def trainProjectByName(self, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
        if theName in self.preppedTablesDF:
            self.trainedModels[theName] = tm.trainModels(theName, self)
            self.trainedModels[theName].fitModels()
        return

    """
    Purpose:
            
    Call:
            
    Example:
            
    """
     def prepProjectByBatch(self):
        for tableName in self.batchTablesList:
            if tableName in self.preppedTablesDF:    
                self.preppedData[tableName] = prepData(tableName, self)
        return
 
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
     def trainProjectByBatch(self):
        for tableName in self.batchTablesList:
            if tableName in self.preppedTablesDF:    
                self.trainedModels[tableName] = tm.trainModels( tableName, self)
                self.trainedModels[tableName].fitModels()
        return
  
    
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
    def exportBestModel(self, filename, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
        predict = predictProject(self, theName, self.bestModelName)
        predict.exportPredictClass(filename)

    """
    Purpose:
            
    Call:
            
    Example:
            
    """
 
    def createPredictFromBestModel(self, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
        return predictProject(self, theName, self.bestModelName)
        
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
    def createPredictFromNamedModel(self, namedModel, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
        
        return predictProject(self, theName, namedModel)


    """
    Purpose:
            
    Call:
            
    Example:
            
    """
    def exportNamedModel(self, namedModel, filename, tableName=None):
        if tableName is not None:
            theName = tableName
        else:
            theName = self.defaultPreppedTableName
        
        predict = predictProject(self, theName, namedModel)
        predict.exportPredictClass(filename)
        
    """
    Purpose:
            
    Call:
            
    Example:
            
    """  
    def addManualRuleForTableName(self, tableName, functionName, columnName, value ): 
        if tableName in self.preppedTablesDF:
            df = self.preppedTablesDF[tableName]
            self.cleaningRules[tableName].addManualRule(functionName, columnName, value, df)
    
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
    def addManualRuleForDefault(self, functionName, columnName=None, value=None ):
        if self.defaultPreppedTableName in self.preppedTablesDF:
            df = self.preppedTablesDF[self.defaultPreppedTableName]
            self.cleaningRules[self.defaultPreppedTableName].addManualRule(functionName, columnName, value, df)


    """
    Purpose:
            
    Call:
            
    Example:
            project.setGoals({'AUROC':(0.70,'>'),'Precision':(0.386,'>'),'fbeta':(0.44,'>')})
    """
 
    def setGoals(self, goals):
        self.goalsToReach = goals
        return



    """
    project.setOngoingReporting(True,'Loan Data')

    """
    def setOngoingReporting(self, flag, fileName):
        self.ongoingReporting = flag
        self.ongoingReportingFilename = fileName
        return


    """

     project.displayAllScores('Loan Data')
    
     def displayAllScores(self, fileName):

    """
    def displayAllScores(self, fileName, short=False):
        print ('\nReport on file: {}'.format(fileName))
        model = self.trainedModels[fileName]
        
        
        cols = ['Model'] + model.shortModelScoresColumns 
        if self.setGoals is not None:
            cols += ['Goals']

        lst = []
        for r in model.modelScores:
            row = model.modelScores[r]
            
            # Round Values
            for c in row:
                if row[c] is not None:
                    if isinstance(row[c], float):
                        row[c] = round(row[c],3)
            
            row['Model']=r
            
            # Test for goals
            listGoals = ''
            if self.goalsToReach is not None:
                for goal in self.goalsToReach:
                    if goal in row:
                        if row[goal] is not None:
                            val,operand = self.goalsToReach[goal]
                            if operand=='>':
                                if row[goal] > val:
                                    listGoals += '{}:{:5.3f}>{:5.3f} '.format(goal,row[goal], val)
                            elif operand=='<':
                                if row[goal] < val:
                                        listGoals += '{}:{:5.3f}<{:5.3f} '.format(goal,model.row[goal], val)
                            elif operand=='=':
                                if (row[goal]+.05 <= val) and (row[goal]-.05 >= val):
                                        listGoals += '{}:{:5.3f}={:5.3f} '.format(goal,model.row[goal], val)
                        
                row['Goals'] = listGoals
            
            
            lst.append(row)

        utility.printAsTable(lst,columns=cols)
        
        # Message the goals
        msg = '    ** Project Goals: '
        if self.goalsToReach is not None:
            for goal in self.goalsToReach:
                val,operand = self.goalsToReach[goal]
                msg+='{} {} {:5.3f}, '.format(goal,operand, val)
        print (msg)
        
        
        if not short:
            print ('\n')
            print ('   Confusion           Predicted')
            print ('   Matrix:       Negative    Positive')
            print ('              +-----------+-----------+')
            print ('   Actual Neg | True Neg  | False Pos | ')
            print ('   Actual Pos | False Neg | True Pos  |<--Recall = True Pos / (True Pos + False Neg)')
            print ('              +-----------+-----------+          = How many true were actually true')
            print ('                                ^ Precision = True Pos / (False Pos + True Pos) ')
            print ('                                          = How many did we predict correcly\n')
            print ()
            print ('   Accuracy = how many out of the total did we predict correctly')
            print ('   F1 Score  = 2 * (Precision * recall) / (Precision + recall)  (1.0 is perfect precision and recall)')
            print ('   f-Beta = F1 score factored 1=best, 0=worst. β<1 favors precision, β>1 favors recall. β->0 only precision, β->inf only recall')
            print ('   MSE (Mean squared error) - distance from the fit line (Smaller the value better the fit)')
            print ('   R2 Compare model to simple model. Ratio of errors of MSE/Simple Model.  Score close to 1=Good, 0=Bad')
            print ('   AUROC area under curve of true positives to false positives. Closer to 1 is better')
   

    """
    Purpose:
            
    Call:
            
    Example:
            
    """   
    def reportResultsOnTrainedModel(self, fileName, modelName):
        print ('\nReport on model: ', modelName)
        
        
        model = self.trainedModels[fileName].fittedModels[modelName]
        scores = self.trainedModels[fileName].modelScores[modelName]
         
    
        if scores['roc_curve'] is not None:
            fpr, tpr, threshold = scores['roc_curve']
        else:
            fpr, tpr, threshold = None, None
        confusionMatrix = scores['CM']
        
        
        for s in self.trainedModels[fileName].shortModelScoresColumns:
            print ('  {} = {}'.format(s,scores[s]))
        print ('  Confusion Matrix = {}\n  DataShape = {}\n'.format(confusionMatrix, self.preppedTablesDF[fileName].shape))
        
        
        print ('\nModel details:\n')
        print (model)
        print ('\nModel Best Params:\n')
        print (model.best_params_)
        print ()
        
        if fpr is not None:
            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

            ax[0].plot(threshold, tpr + (1 - fpr))
            ax[0].set_xlabel('Threshold')
            ax[0].set_ylabel('Sensitivity + Specificity')

            ax[1].plot(threshold, tpr, label="tpr")
            ax[1].plot(threshold, 1 - fpr, label="1 - fpr")
            ax[1].legend()
            ax[1].set_xlabel('Threshold')
            ax[1].set_ylabel('True Positive & False Positive Rates')
            plt.show()
            
            function = tpr + (1 - fpr)
            index = np.argmax(function)

            optimalThreshold = threshold[np.argmax(function)]
            print ('Optimal Threshold:', optimalThreshold)
            print ()
            print ()
            
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % scores['AUROC'])
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
        
        if confusionMatrix is not None:
            # Plot non-normalized confusion matrix
            
            if self.confusionMatrixLabels is not None:
                classes = []
                for val, desc in self.confusionMatrixLabels:
                    classes.append('{}({})'.format(desc,val))
                
                
                plt.figure()
                plot_confusion_matrix(confusionMatrix, classes=classes, normalize=False,
                                  title='Confusion matrix, without normalization')
                plt.show()

                # Plot normalized confusion matrix
                #plt.figure()
                #plot_confusion_matrix(confusionMatrix, classes=classes, normalize=True,
                #                  title='Normalized confusion matrix')


    """
    Purpose:
            
    Call:
            
    Example:
            
    """
 
    def logTrainingResults(self, fileName, outputFileName, inputModelName=None):
       print ('\nLogging Results: ')
       
       results = []
       
       modelNames = {'gbc':'gradientboostingclassifier__',
                     'l1':'logisticregression__',
                     'l2':'logisticregression__',
                     'rfc':'randomforestclassifier__',
                     'bagging':'baggingclassifier__',
                     'adaboost':'adaboostclassifier__',
                     'gaussiannb':'gaussiannb__',
                     'baggingbase':'baggingclassifier__',
                     'adaboostbase':'adaboostclassifier__',
                     'decisiontree':'decisiontreeclassifier__',
                     'kneighbors': 'kneighborsclassifier__',
                     'sgd':'sgdclassifier__'
                   }
       
       hyperparametersToReport = ['loss','max_depth','learning_rate','C','max_iter','solver','max_features',
                                  'n_estimators','max_samples','algorithm','penalty','tol']
       scoresToReport = ['AUROC','fbeta', 'Recall', 'Precision','RunTime', 'F1', 'Accuracy', 'MAE', 'r2']
    
       header = 'Model'
       for report in scoresToReport:
           header += ', {}'.format(report)
       for param in hyperparametersToReport:
               header += ', {}'.format(param)
       header += ', runParams'
       #print ('\n')
       #print (header)
       header+= '\n'
       
       # Check is the file exists and than open for write or append
       myFile = Path(outputFileName)
       if myFile.is_file():
           file = open(outputFileName,'a')
       else:
           file = open(outputFileName,'w')
           file.write(header)
    
    
       # Determine if the model is to report on all or just part.
       if inputModelName is None:
           modelListToProcess = self.modelListAsRun
       else:
           modelListToProcess = [inputModelName]
           
       for modelName in modelListToProcess:
       
           model = self.trainedModels[fileName].fittedModels[modelName]
           scores = self.trainedModels[fileName].modelScores[modelName]
           
               
           row = '{}'.format(modelName)
           for report in scoresToReport:
               if scores[report] is None:
                   row += ', None'
               else:
                   row += ', {:5.3f}'.format(scores[report])
        
        
           for param in hyperparametersToReport:
               lookup = modelNames[utility.removePlus(modelName)]+param
               #print ('lookup=',lookup)
               if lookup in model.best_params_ :
                   row += ', {}'.format(model.best_params_[lookup])
               else:
                   row += ','

           hp = self.trainedModels[fileName].hyperparameters[modelName]
           hpStr = '{'
           for h in hp:
               hpStr+='{}: {},'.format(h,hp[h])
           hpStr += '}'
           #hpFixed = ''
           #for c in hpStr:
           #    if c=='"':
           #        hpFixed+=
            #   else:
            #       hpFixed+=c
           row += ',"{}"'.format(hpStr)
           print (row)
           row += '\n'
           file.write(row)
           
       file.close()
       #print ('\n')
       #for x in model.best_params_ :
       #    print (x)
   
 
"""
    Purpose:
            
    Call:
            
    Example:
            
"""
 
class predictProject (object):
    
    def __init__ (self, project, tableName=None, namedModel=None):
        self.name = project.name
        self.description = project.description
        self.modelType = project.modelType
        self.cleaningRules = None
        self.predictFile = None
        self.predictDataDF = None
        self.predictSet = None
        self.readyToRun = False

        
        if tableName is not None:
            if tableName in project.preppedTablesDF:    
                theName = tableName
            else:
                theName = project.defaultPreppedTableName
        else:
            theName = project.defaultPreppedTableName
        
        if namedModel:
            name = namedModel
        else:
            name = project.bestModelName


        theTrainedModel = project.trainedModels[theName]
        if theName in project.cleaningRules:
            self.cleaningRules = project.cleaningRules[theName]
                   

        if name in theTrainedModel.fittedModels:
            self.modelName = name
            self.modelScore = theTrainedModel.modelScores[name]
            if self.modelType == tm.TRAIN_CLUSTERING:
                self.model = theTrainedModel.fittedModels[name]
            else:
                self.model = theTrainedModel.fittedModels[name].best_estimator_
        else:
            utility.raiseError('project name,{}, not found'.format(name))
            
        
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
         
    def importPredictFile(self, name, type=None, description=None, location=None, fileName=None, sheetName=None, hasHeaders = False, range=None):
        self.predictFile = getData(name, type=type, description=description, location = location, fileName=fileName,  sheetName=sheetName, range=range, hasHeaders = hasHeaders)
        self.predictDataDF = self.predictFile.openTable()


    """
    Purpose:
            
    Call:
            
    Example:
            
    """
    def importPredictFileFromProject(self, project, tableName):
        if tableName in project.preppedTablesDF:
            self.predictDataDF = project.preppedTablesDF[tableName]

    """
    Purpose:
            
    Call:
            
    Example:
            
    """
  
    def importPredictFromDF(self, df):
        self.predictDataDF = df
  
 """
 Purpose:
         
 Call:
         
 Example:
         
 """
    
    def prepPredict(self):
        prep = prepPredictData(self)
        self.predictSet = prep.getPredictSet()
        self.readyToRun = True
        
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
         
    def exportPredictClass(self, filename):
        with open(filename, 'wb') as f:
            pk.dump(self, f)

    """
    Purpose:
            
    Call:
            
    Example:
            
    """
 
    def addToPredictFile(self, columnName, columnData):
        self.predictDataDF[columnName] = columnData

    """
    Purpose:
            
    Call:
            
    Example:
            
    """
 
    def exportPredictFile(self, filename):
        if self.predictDataDF is not None:
            self.predictDataDF.to_csv(filename, index=False)

                   
    """
    Purpose:
            
    Call:
            
    Example:
            
    """
             
    def runPredict(self):
        try:
            if self.readyToRun:
                if self.modelType == tm.TRAIN_CLASSIFICATION:
                    
                    pred = self.model.predict(self.predictSet)
                    #p = self.model.predict_proba(self.predictSet)
                    #pred = [x[1] for x in p]
                else:
                    pred = self.model.predict(self.predictSet)
                return pred
            return None
        
        except NotFittedError as e:
            print (repr(e))

    """
    Purpose:
            
    Call:
            
    Example:
            
    """
 
def loadPredictProject(filename):
        with open(filename, 'rb') as f:
            return pk.load(f)
            

"""
    Purpose:
            
    Call:
            
    Example:
            
"""
 
#
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #cm = cm.astype('float')/cm.sum(axis=0)
        #print ("Normalized confusion matrix")
    else:
        #print ('Confusion matrix, without normalization')
        pass

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

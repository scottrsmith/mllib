#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from mlLib.project import mlProject, loadPredictProject
import mlLib.exploreData as ed
import mlLib.trainModels as tm
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)



project = mlProject('Titanic Survival', 'Udacity')

#modelList=['bagging']
#modelList=['l1', 'l2', 'rfc', 'decisiontree', 'kneighbors', 'bagging', 'adaboost', 'baggingbase', 'adaboostbase', 'gaussiannb']
modelList=['bagging']
#modelList=None  #defauklt is All
project.setTrainingPreferences (crossValidationSplits=10, parallelJobs=-1, modelType=tm.TRAIN_CLASSIFICATION, modelList=modelList, 
                                useStandardScaler = True,fbeta=None, defaultHyperparameters=None, hyperparametersLongRun=True)


#project.openProject('Predict Home Values2', type='csv', description='Real Estate Project 2', fileName='original_real_estate_data.csv',  hasHeaders = True)
project.importFile('Survival Data', type='csv', description='Survival Data', fileName='titanic_train.csv',  hasHeaders = True)


project.setTarget('Survived')



project.exploreData() 
   


print (project.explore['Survival Data'])
print (project.explore['Survival Data'].allStatsSummary())
project.explore['Survival Data'].plotExploreHeatMap()
project.explore['Survival Data'].plotHistogramsAll(10)
project.explore['Survival Data'].plotCorrelations()



#########
# Feature Engineer
#
#project.addManualRuleForDefault(ed.CLEANDATA_REMOVE_ITEMS_EQUAL, 'CustomerID', None)
#project.addManualRuleForDefault(ed.CLEANDATA_CONVERT_DATATYPE, 'CustomerID', 'int64')
#project.addManualRuleForDefault(ed.CLEANDATA_NEW_FEATURE, 'Sales', 'Quantity * UnitPrice')
#project.addManualRuleForDefault(ed.CLEANDATA_REMOVE_ITEMS_EQUAL, 'department', 'temp')
#project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET, 'department', [['information_technology'], 'IT'])
#project.addManualRuleForDefault(ed.CLEANDATA_NUMERIC_MARK_MISSING, 'last_evaluation', None)
#project.addManualRuleForDefault(ed.CLEANDATA_DROP_NA)
#project.addManualRuleForDefault(ed.CLEANDATA_REMOVE_ITEMS_BELOW, 'property_age',-1)
#project.addManualRuleForDefault(ed.CLEANDATA_DROP_COLUMN,'tx_year', None)


project.initCleaningRules()
project.addManualRuleForDefault(ed.CLEANDATA_NUMERIC_MARK_MISSING, 'Age', None)
project.addManualRuleForDefault(ed.CLEANDATA_ZERO_FILL, 'Fare')
project.addManualRuleForDefault(ed.CLEANDATA_DROP_COLUMN,['Passengerid','Name','Ticket','Cabin'], None)
project.addManualRuleForDefault(ed.CLEANDATA_DROP_NA)


project.cleanAndExploreProject()
#
project.prepProjectByName('Survival Data')
project.trainProjectByName('Survival Data')



print 
print ('The best is ', project.bestModelName)
print
print (project.bestModel)


project.displayAllScores('Survival Data')
project.reportResultsOnTrainedModel('Survival Data',project.bestModelName)


#
predict = project.createPredictFromBestModel('Survival Data')
predict.importPredictFile('Kaggle Data', type='csv', description='Raw Data', fileName='titanic_test.csv',  hasHeaders = True)
predict.prepPredict()
ans = predict.runPredict()
predict.addToPredictFile('Survived',ans)
predict.removeFromPredictFile(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Age_missing'])
predict.exportPredictFile('kaggleSubmit.csv')

#print (ans)

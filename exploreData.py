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

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd


# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
#%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns
import mlLib.mlUtility as mlUtility

#from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel




"""

Explore the data - what decisions have to be made to get ready for the clean"

1. What columns have null data
2. what columns have duplicates
3. What columns are binary data
4. what categores need:
    to fix capitalization
    to regroup / combine
    
     

5. What columns need outliers dropped

6. what columns have missing features




"""


CLEANDATA_MARK_MISSING = 'Mark Missing'
CLEANDATA_REBUCKET = 'Rebucket'
CLEANDATA_FIX_SKEW = 'Fix skew via log function'
CLEANDATA_REBUCKET_WHERE = 'Rebucket column from 2nd column where'
CLEANDATA_REBUCKET_TO_BINARY = 'Rebucket to Binary'
CLEANDATA_REBUCKET_BY_RANGE = 'Rebucket by Range'
CLEANDATA_REBUCKET_BY_INCLUDE = 'Rebucket by Included String'
CLEANDATA_FIX_CASE = 'Fix Case'
CLEANDATA_ZERO_FILL = 'Zero Fill'
CLEANDATA_REMOVE_ITEMS_BELOW = 'Remove Items <='
CLEANDATA_REMOVE_ITEMS_ABOVE = 'Remove Items >='
CLEANDATA_NEW_FEATURE = 'New Feature'
CLEANDATA_DROP_COLUMN = 'Drop Column'
CLEANDATA_DROP_NO_CORR_COLUMN = 'Drop Column with Low Correlation'
CLEANDATA_NEW_INDICATOR_VARIABLE = 'New Indicator Variable'
CLEANDATA_REMOVE_ITEMS_EQUAL = 'Remove Items Equal'
CLEANDATA_KEEP_ITEMS_EQUAL = 'Keep Items Equal'
CLEANDATA_NUMERIC_MARK_MISSING = 'Numeric Mark Missing'
CLEANDATA_DROP_DUPLICATES = 'Drop Duplicates'
CLEANDATA_CONVERT_DATATYPE = 'Convert Datatype'
CLEANDATA_CONVERT_TO_BOOLEAN = 'Convert Datatype to Boolean'
CLEANDATA_GROUPBY_ROLLUP = 'Group-by Rollup'
CLEANDATA_UPDATE_FROM_TABLE2 = 'Update training table from second table'
CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP =  'Intermediary Group-by Rollup'
CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP = 'Rollup Intermediary Group-by Rollup'
CLEANDATA_JOIN_ROLLUPS = 'Join Group-by Rollups'
CLEANDATA_DIMENSIONALITY_REDUCTION = 'Explore Dimensionality Reduction'
CLEANDATA_PCA_DIMENSIONALITY_REDUCTION = 'PCA Dimensionality Reduction'
CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION =  'Threshold Dimensionality Reduction'
CLEANDATA_JOIN_TABLES = 'Join tables'
CLEANDATA_STRIP_TO_ALPHA = 'Strip to alpha. None='
CLEANDATA_SET_BATCH_TABLES = 'Set table to run in batch'
CLEANDATA_DROP_NA = 'Drop rows with missing values'
CLEANDATA_DROP_NA_FOR_COLUMN = 'Drop rows with missing values in column'
CLEANDATA_SET_CATEGORY_DATATYPE = 'Convert object type to category'
CLEANDATA_NEW_FEATURE_DATE_DIFFERENCE_MONTHS = 'New feature date difference in months'


CLEANDATA_CONVERT_CATEGORY_TO_INTEGER = 'Convert Category to Integer'
CLEANDATA_CONVERT_COLUMN_TO_INTEGER = 'Convert Column to Integer'
CLEANDATA_DROP_ID_FIELD = 'Identified list of uniqie values. Dropped as an ID field'

CLEANDATA_SPARSE_SERIES = 'This feature has a sparse score of'
CLEANDATA_UNBALANCED_DATA = 'This feature seems to be unbalanced'
CLEANDATA_SKEWED_DATA = 'The distribution of this data may be skewed'
CLEANDATA_HIGH_CORRELATION = 'There is a high correlation'
CLEANDATA_HIGH_FEATURE_IMPORTANCE = 'There is a high feature importance'
CLEANDATA_DROP_LOW_FEATURE_IMPORTANCE_AND_CORR = 'Dropping Column with low correlation and feature importance'

CLEANDATA_NO_OPP = 'No Operation'

exploreAutoFunctions = [
                        CLEANDATA_DROP_DUPLICATES,
                        CLEANDATA_MARK_MISSING,
                        CLEANDATA_NUMERIC_MARK_MISSING,
                        CLEANDATA_FIX_CASE,
                        CLEANDATA_CONVERT_CATEGORY_TO_INTEGER,
                        CLEANDATA_CONVERT_COLUMN_TO_INTEGER,
                        CLEANDATA_ZERO_FILL,
                        CLEANDATA_DROP_ID_FIELD,
                        CLEANDATA_DROP_COLUMN,
                        CLEANDATA_DROP_NO_CORR_COLUMN,
                        CLEANDATA_SKEWED_DATA,
                        CLEANDATA_UNBALANCED_DATA,
                        CLEANDATA_DROP_LOW_FEATURE_IMPORTANCE_AND_CORR,
                        CLEANDATA_SET_CATEGORY_DATATYPE
                        ]



exploreHeatMap = [CLEANDATA_MARK_MISSING, 
                  CLEANDATA_FIX_CASE,
                  CLEANDATA_ZERO_FILL,
                  CLEANDATA_DROP_DUPLICATES,
                  CLEANDATA_REBUCKET,
                  CLEANDATA_REMOVE_ITEMS_BELOW,
                  CLEANDATA_REMOVE_ITEMS_ABOVE,
                  CLEANDATA_NEW_FEATURE,
                  CLEANDATA_DROP_COLUMN,
                  CLEANDATA_NEW_INDICATOR_VARIABLE,
                  CLEANDATA_REMOVE_ITEMS_EQUAL,
                  CLEANDATA_NUMERIC_MARK_MISSING,
                  CLEANDATA_CONVERT_DATATYPE,
                  CLEANDATA_DIMENSIONALITY_REDUCTION,
                  CLEANDATA_SPARSE_SERIES,
                  CLEANDATA_UNBALANCED_DATA,
                  CLEANDATA_SKEWED_DATA,
                  CLEANDATA_HIGH_CORRELATION,
                  CLEANDATA_HIGH_FEATURE_IMPORTANCE,
                  CLEANDATA_DROP_NO_CORR_COLUMN,
                  CLEANDATA_CONVERT_CATEGORY_TO_INTEGER,
                  CLEANDATA_CONVERT_COLUMN_TO_INTEGER,
                  CLEANDATA_DROP_LOW_FEATURE_IMPORTANCE_AND_CORR
                  ]
  
CLEANDATA_BREAK = '|'
 





def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh   



# from http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
def doubleMADsfromMedian(y,thresh=3.5):
    # warning: this function does not check for NAs
    # nor does it address issues when 
    # more than 50% of your data have identical values
    try:
        m = np.median(y)
        abs_dev = np.abs(y - m)
        left_mad = np.median(abs_dev[y <= m])
        right_mad = np.median(abs_dev[y >= m])
        y_mad = left_mad * np.ones(len(y))
        y_mad[y > m] = right_mad
       
        modified_z_score = 0.6745 * abs_dev / y_mad
        #    modified_z_score = abs_dev / y_mad
        modified_z_score[y == m] = 0
        return modified_z_score > thresh
        
        
    except Exception as e:
        raise e
    except RuntimeWarning:
        return []
   

    
def getHistogram(series):
    hist = {}
    for value in series:
        if value in hist:
            hist[value] += 1
        else:
            hist[value]  = 1
    return hist


def getStats (series):
    
    s = series.describe()  
    stats = {}
    
    # Stats for all
    stats['count'] = s['count']
    if hasattr(series, 'cat'):
        stats['dataType'] = 'category'
    else:
        stats['dataType'] = series.dtype
    #mlUtility. traceLog(('dtype=', series.dtype)
    
    # Null Count
    stats['nullCount'] = series.isnull().sum()
    
    
    stats['mean'] = None
    stats['median'] = None
    stats['std'] = None 
    stats['min'] = None
    stats['1%'] = None
    stats['2.5%'] = None
    stats['5%'] = None
    stats['25%'] = None
    stats['50%'] = None
    stats['75%'] = None
    stats['97.5%'] = None
    stats['95%'] = None
    stats['99%'] = None
    stats['max'] = None        
    stats['first'] = None
    stats['last']  = None  
    stats['top'] = None
    stats['freq'] = None
   
    
 
    stats['histogram'] = getHistogram(series)
    # Status for Objects
    if series.dtype == 'object':
        stats['unique'] = s['unique']
        stats['top'] = s['top']
        stats['freq'] = s['freq']

    elif hasattr(series,'cat'):
        stats['unique'] = s['unique']
        stats['top'] = s['top']
        stats['freq'] = s['freq']
        
    # stats for dates
#    elif hasattr(series, 'datetime'):
    elif series.dtype == 'datetime64[ns]':
        stats['unique'] = s['unique']
        stats['top'] = s['top']       
        stats['freq'] = s['freq']      
        stats['first'] = s['first']    
        stats['last'] = s['last']    

    elif series.dtype == 'bool':
        stats['unique'] = s['unique']
        stats['unique'] = len(series.unique())


    else:
        # Status for numbers
        stats['mean'] = s['mean']
        stats['median'] = np.median(series)
        stats['std'] = s['std'] 
        stats['min'] = s['min']
        stats['25%'] = s['25%']
        stats['50%'] = s['50%']
        stats['75%'] = s['75%']
        stats['max'] = s['max']        
        q = series.quantile([.01, 0.025, 0.05, 0.95, 0.975, 0.99])
        stats['1%'], stats['2.5%'], stats['5%'], stats['95%'], stats['97.5%'], stats['99%'] = q[0.01], q[0.025], q[0.05], q[0.95], q[0.975], q[0.99] 
         
        stats['unique'] = len(series.unique())
        stats['top'] = None
        stats['freq'] = 0
        hist = stats['histogram']
        for value in hist:
            if hist[value] >= stats['freq']:
                stats['freq'] = hist[value]
                stats['top'] = value
         
   
    # test for Binary
    if stats['unique'] == 2 and series.dtype != 'object':
        stats['binary']=True
    else:
        stats['binary']=False
#    print ('\n\nStats')
#    print (stats)

#    print ('\n\ns')
#    print (s)


    return stats
    

def runExploreTest(function, project, explore, df, dfCopy, colName, stats):
    rules, dropped = function(project, explore, df, dfCopy, colName, stats)
    if rules is not None:
        # Run the Rules for the column
        for r in rules:
            r.execute(df, project, None) 
        dfCopy[colName] = df[colName]
        
    return rules, dropped
    
    
def addExploreRules(rules, name, explore, value=None, key=None):
       
    if key is not None :
        rules[explore+CLEANDATA_BREAK+key] = value
    else:
        rules[explore] = value
    return 

    
#columnTests = {'object': [ testIDField, ],
#                 'number': [],
#                 'bool': [],
#                 'date': [] 
#                }
    

def testIDField(project, explore, df, dfCopy, colName, stats):
    rules = {}
    dropped = False
    # Do the tests
    if stats['unique'] >= stats['count'] * .90:
        # Add tge ryes
        # this is an ID field of uniqie values, needs to be dropped
        addExploreRules(rules, name,CLEANDATA_DROP_ID_FIELD, None)
        dropped = True
    return rules, dropped
   
    

class exploreData(object):
    
    def __init__(self, df, project, name):
        
       
        self.columns = {}
        self.recommendations = {}
        self.colsToDrop = []
        self.dataFrame = df
        self.project = project
        self.fileName = name
        #self.featureImportance = None
        #self.correlations = None
        if name in project.targetVariable:
            self.target = project.targetVariable[name]
        else:
            self.target = None
        self.topColumns = None
        self.correlations = None
        self.featureImportance = None
        
        
        # Init explore heatmap to zeros
        self.heatMap = pd.DataFrame(np.zeros([len(exploreHeatMap),len(df.columns)], dtype=float), index=exploreHeatMap, columns=df.columns)

        
        d  = np.zeros([len(exploreHeatMap),len(df.columns)], dtype=float)
        self.headmap = pd.DataFrame(d, index=exploreHeatMap, columns=df.columns)

        

        print ('Evaluating correlations and feature importance\n')
        
        # make a copy of the data to do numerical evaluations
        dfCopy = df.copy()
        #dfCopy.to_csv('CorrFI-TestData-before.csv')
        if self.target is not None:
            dfCopy[self.target].dropna()
        if project.IsTrainingSet in dfCopy:
           dfCopy.drop(project.IsTrainingSet, axis=1,  inplace=True)

        for name in dfCopy:
            type = dfCopy.dtypes[name]                      
            if type == 'object' or hasattr(dfCopy[name], 'cat'):
                dfCopy[name], _ = pd.factorize(dfCopy[name])
                dfCopy[name] = dfCopy[name].astype('int64')
            elif type == 'datetime64[ns]':
                dfCopy[name], _ = pd.factorize(dfCopy[name])
                dfCopy[name] = dfCopy[name].astype('int64')
            dfCopy[name].fillna(-np.inf, inplace = True)
            
            
        evaluationColumns = dfCopy.columns                
        # for each column, now use the best freature engineering score    
        
        #           
        if self.target is not None:
             self.correlations = self.calcCorrelations(dfCopy)
             self.featureImportance = self.calcFeatureImportance(dfCopy)
        #dfCopy.to_csv('CorrFI-TestData.csv')
    
        
        del dfCopy

        print ('Exploring Data', end='')
        
        for name in df:
            print ('.', end='')
            stats = getStats(df[name])
            self.columns[name] = stats
            #print ('name in ', name, stats['dataType'])
            #mlUtility.runLog('\nStat for=',name)
            #mlUtility.runLog(self.columns[name])
            if name != self.target:
                if stats['dataType'] == 'object':
                    self.reviewObjects(name, stats, df[name], project)
                elif hasattr(self.dataFrame[name],'cat'):
                    self.reviewObjects(name, stats, df[name], project)
                elif stats['dataType'] == 'datetime64':
                    self.reviewDatetime(name, stats, df[name])
                elif stats['dataType'] == 'bool':
                    self.reviewBool(name, stats, df[name])
                else:
                    self.reviewNumbers(name, stats, df[name])  
        print ()
        if self.target is not None:
            self.evaluateColumnImportance(evaluationColumns, self.correlations, self.featureImportance, doDrop=True)
        return
#        try:

        
#        except AssertionError:
#            raise Exception(self.KAT_TABLE_ERROR)
            
#        except Exception as e:
#            raise e
  
  

  
    def featureTesting(self, project, df):
        print ('Evaluating correlations and feature importance\n')
        
        # make a copy of the data to do numerical evaluations
        dfCopy = df.copy()
        #dfCopy.to_csv('CorrFI-TestData-before.csv')
        
        # Build the training data into floats
        if self.target is not None:
            dfCopy[self.target].dropna()
        if project.IsTrainingSet in dfCopy:
           dfCopy.drop(project.IsTrainingSet, axis=1,  inplace=True)

        for name in dfCopy:
            type = dfCopy.dtypes[name]                      
            if type == 'object' or hasattr(dfCopy[name], 'cat'):
                dfCopy[name], _ = pd.factorize(dfCopy[name])
                dfCopy[name] = dfCopy[name].astype('int64')
            elif type == 'datetime64[ns]':
                dfCopy[name], _ = pd.factorize(dfCopy[name])
                dfCopy[name] = dfCopy[name].astype('int64')
            dfCopy[name].fillna(-np.inf, inplace = True)
        
        
        # Get the initial scores for each column
        correlations = self.calcCorrelations(dfCopy)
        featureImportance = self.calcFeatureImportance(dfCopy)

        # Seed the initial values
        for colName in df:
            if colName != self.target:
                self.bestScores[colName] = featureImportance[colName] + abs(correlations[self.target][colName])
                self.bestRules[colName] = []

        # for each column in the source file 
        bestScores = {}
        bestRules = {}
        for colName in df:
            print ('.', end='')
            stats = getStats(df[colName])
            self.columns[colName] = stats
            if colName != self.target:
                # for each column transformation test
                datatype = stats['dataType']
                saveCol = dfCopy[colName]
                if stats['dataType'] == 'object':
                    tests = columnTests[datatype]
                elif hasattr(self.dataFrame[name],'cat'):
                    tests = columnTests['object']
                elif stats['dataType'] == 'datetime64':
                    tests = columnTests['date']
                elif stats['dataType'] == 'bool':
                    tests = columnTests['bool']
                else:
                    tests = columnTests['number']
                testingColumn = df[colName]
                for functionName in tests:
                    rules, dropped = runExploreTest(functionName,self.project, self, testingColumn, dfCopy, colName, stats)
                    if dropped:
                        bestScores[colName] = 999
                        bestRules[colName] = rules
                        break                       
                    if rules is not None:
                        correlations = self.calcCorrelations(dfCopy)
                        featureImportance = self.calcFeatureImportance(dfCopy)
                        score = featureImportance[colName] + abs(correlations[self.target][colName])
                        if bestScores[colName] < score:
                            bestScores[colName] = score
                            bestRules[colName] = rules
                dfCopy[colName] = saveCol
        # Get the best rules
        for colName in self.bestRules:
           self.recommendations[colName] = bestRules[colName]
  
  
    def __str__ (self):
        str = ''
        for key, value in self.recommendations.items():
            str += '\nRecommendations for column {}, type {}\n'.format(key, self.columns[key]['dataType'])
            for k, v in value.items():
                str+= '  --->{} value={}\n'.format(k,v)
        return str
    
    
    def __getitem__ (self, name):
        if name in self.columns:
            if name in self.recommendations:
                return self.columns[name], self.recommendations[name]
        return None

        
    def statsSummary (self, name):
        if name in self.columns:     
            stats = self.columns[name]
            dStats = {}
            for s in stats:
                if stats[s] is None:
                    dStats[s] = 'None'
                else:
                    dStats[s] = stats[s]
             
            str = 'Stats for Column: {}  datatype={}\n'.format(name, dStats['dataType'])
            str += '   mean:   {:<12.4}   first:  {}\n'.format(dStats['mean'] ,dStats['first'])
            str += '   median: {:<12}   last:   {}\n'.format(dStats['median'], dStats['last'])
            str += '   std:    {:<12.4}   top:    {}\n'.format(dStats['std'], dStats['top'])
            str += '   min:    {:<12}   freq:   {}\n'.format(dStats['min'], dStats['freq']   )
            str += '   max:    {:<12}   uniqie: {:<12}\n'.format(dStats['max'],dStats['unique'] )
            str += ' count:    {:<12}   nulls:  {:<12}\n\n'.format(dStats['count'],dStats['nullCount'] )
#            str += '    1%:    {:<10}   2.5%:   {:<10}   5%:   {:<10}      25%:   {:<10}    50%:   {:<10}\n'.format(stats['1%'], stats['2.5%'], stats['5%'] , stats['25%'] ,stats['50%'])
#            str += '   50%:    {:<10}   75%:    {:<10}  95%:   {:<10}    97.5%:   {:<10}    99%:   {:<10}\n'.format(stats['50%'], stats['75%'], stats['95%'] , stats['97.5%'] , stats['99%'])
            #mlUtility. traceLog((name)
            if stats['dataType'] == 'bool':
                pass
            elif stats['dataType']=='datetime64':
                pass
            elif hasattr(self.dataFrame[name],'cat'):
                pass
            elif stats['dataType'] != 'object':
                str += '   {:9d}% {:9.1f}% {:9d}% {:9d}% {:9d}% {:9d}% {:9d}% {:9.1f}% {:9d}%\n'.format(1, 2.5, 5, 25, 50, 75, 95, 97.5, 99)
                str += '   {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}\n'.format(stats['1%'], stats['2.5%'], stats['5%'], stats['25%'], stats['50%'], stats['75%'], stats['95%'] , stats['97.5%'] , stats['99%'])
            return str
        return None

    def allStatsSummary (self):
        str = 'List of all Stats\n'
        for name in self.dataFrame.dtypes.index:
            str += self.statsSummary(name)
            str += '\n'
        return str
            

    def dataSummary (self, rows=5):     
        mlUtility.runLog ('\n\nData Summary:')
        mlUtility.runLog ('\n\nFirst {} rows: '.format(rows))
        mlUtility.runLog (self.dataFrame.head(n=rows))
        mlUtility.runLog ('\n\nLast {} rows: '.format(rows))
        mlUtility.runLog (self.dataFrame.tail(n=rows))
        mlUtility.runLog ('\n')
        



    def getColumnValues(self, name):
        if name in self.dataFrame:
            return self.dataFrame[name].unique().tolist()
        return []

        


    def evaluateColumnImportance(self, columns, correlations, featureImportance, doDrop=False):
    
        colsToDrop = self.colsToDrop
        target = self.target
                
    
        col = []
        for name,i in sorted(featureImportance.items(), key=lambda item: item[1]):
            if name not in colsToDrop:
                col.append(name)
    
        # Top 80% of 
        topFI = col[round(len(col) * self.project.bottomImportancePrecentToCut):]
        #print (topFI)
    
        targetCorrABS = abs(correlations[target].copy())
        targetCorr = correlations[target].copy()
        #print ('\n\n Top Corr')
        #print (targetCorr)
    #    for name in targetCorr.index:
        col = []
        for name,_ in sorted(targetCorrABS.items(), key=lambda item: item[1]):
            #c = targetCorr[name]
            if name!=target:
                if name not in colsToDrop:
                    col.append(name)
                    c = targetCorr[name]
                    if c >= self.project.correlationThreshold or c<= -(self.project.correlationThreshold):
                        self.addRecommendation(name,CLEANDATA_HIGH_CORRELATION,c)
                        #print (name,' has high corr =',c)

                
        topCorr = col[round(len(col) * self.project.bottomImportancePrecentToCut):]
        #print (topCorr)
    
        self.topColumns = list(set(topCorr + topFI))
        mlUtility.runLog ('\nTop Columns = {}'.format(self.topColumns))
        # Add to the drop list if columns not included
        for name in columns:
            if name != target:
                if featureImportance[name] > self.project.featureImportanceThreshold:
                    #print (name, 'high feature importance',featureImportance[name])
                    self.addRecommendation(name,CLEANDATA_HIGH_FEATURE_IMPORTANCE,c)
            
                if name not in colsToDrop and doDrop:
                    if name not in self.topColumns:
                        #print ('Drop low importance', name) 
                        self.colsToDrop.append(name)
                        self.addRecommendation(name,CLEANDATA_DROP_LOW_FEATURE_IMPORTANCE_AND_CORR,
                             'FI={}, Corr={}'.format(featureImportance[name],correlations[self.target][name]))
                             

    def reviewObjects (self, name, stats, data, project):
        
        # Determine what type of data this object is.
        # alpha - only letters
        # alpha-umeric - mnumbers and letters
        # is it multiple words
        # Uniqie % = 
        # 
        
    
        # is it a Category
        # is it an ID - 100% unque, 
        # Part number
        # name
        # Description/text
        # 
        
                
        if stats['nullCount'] > 0:
            self.addRecommendation(name,CLEANDATA_MARK_MISSING,None)
            
        # Category   
        # Text Field      
        # ID
    
        # Check for small samples
        histogram = stats['histogram']
        
        if stats['unique'] <= project.uniqueThreshold:
            for item, value in histogram.items():
                if value < project.smallSample:
                    self.addRecommendation(name,CLEANDATA_REBUCKET,item, str(item))
                    
        # chck for large samples
        elif stats['unique'] > project.highDimensionality:
            self.addRecommendation(name,CLEANDATA_DIMENSIONALITY_REDUCTION,stats['unique'])
    
        # check for capitalization
    
        for item in iter(histogram.keys()):
            if item is not np.nan :
                lower = item.lower()
                #mlUtility. traceLog(("item, lower = ",item, lower)
                if item != lower:
                    if lower in histogram:
                        #mlUtility. traceLog(('   --> add recommendations',lower, item)
                        self.addRecommendation(name,CLEANDATA_FIX_CASE,[lower, item], lower+CLEANDATA_BREAK+str(item))
                upper = item.upper()
                if item != upper:
                    if upper in histogram:
                        self.addRecommendation(name,CLEANDATA_FIX_CASE+CLEANDATA_BREAK, [upper, item],
                                     upper+CLEANDATA_BREAK+str(item))
    
        # Figure out how uniqie the values are and make a recommendation
        if stats['unique'] >= stats['count'] * .90:
            # this is an ID field of uniqie values, needs to be dropped
            self.addRecommendation(name,CLEANDATA_DROP_ID_FIELD)
            self.colsToDrop.append(name)
        elif stats['unique'] <= stats['count'] * .10:
            # do nothing. can use one-hot encoding
            self.addRecommendation(name,CLEANDATA_SET_CATEGORY_DATATYPE,None)
        elif stats['unique'] <= stats['count'] * .50:
            # factorize!
            self.addRecommendation(name,CLEANDATA_CONVERT_CATEGORY_TO_INTEGER)
        else:
            self.addRecommendation(name,CLEANDATA_CONVERT_COLUMN_TO_INTEGER) 
            
        
        # Determine if a key
        pass
        
        # Check for unbalanced
        self.checkForImbalanced (name, stats, data, isObject=True)
                
        return 
    
        


    def addRecommendation (self, name, explore, value=None, key=None):
        if name in self.recommendations:
            nameRecs = self.recommendations[name]
        else:
            nameRecs = {}
            self.recommendations[name] = nameRecs
            
        if key is not None :
            nameRecs[explore+CLEANDATA_BREAK+key] = value
        else:
            nameRecs[explore] = value
        
        if explore in exploreHeatMap:
            self.heatMap.at[explore, name] += 1
   


    def calcFeatureImportance(self, df):
        from xgboost import XGBClassifier
        
        
        target = self.project.targetVariable[self.project.defaultPreppedTableName]
                
        # Create separate object for input features
        y = df[target]

   
        # Create separate object for target variable
        df.drop(target, axis = 1, inplace=True)
        col = df.columns

        # Build a forest and compute the feature importances
        forest = XGBClassifier(random_state=self.project.randomState, n_jobs=-1)

        forest.fit(df, y)
        importances = forest.feature_importances_
        #indices = np.argsort(importances)[::-1]
        
        importance = {}
        for n,i in zip(df.columns,importances):
            importance[n] = i
        
        return importance

        

    def calcCorrelations(self, df):
        return df.corr() 



    def checkForSparse(self, name, stats, series):
        
        sparseScore = stats['nullCount'] / len(series)
        if  sparseScore > .5:
            self.addRecommendation(name,CLEANDATA_SPARSE_SERIES,sparseScore)


  
    
    def reviewNumbers (self, name, stats, series):
        # is it a Category?
        # is it an ID - 100% unque, 
        # Part number (repeats)
        # name
        # Description/text
        # skewed
        # Normal distribution
        # 
         
        #print (name)
        if stats['nullCount'] > 0:
            if stats['binary']==True:
                self.addRecommendation(name,CLEANDATA_ZERO_FILL,name)
            else:
                self.addRecommendation(name,CLEANDATA_ZERO_FILL,'mean')
        
        #recommendations['Remove the top 1% for '+ name] =  stats['99%']
        #recommendations['Remove the bottom 1% for '+ name] =  stats['1%']
        
        self.checkforOutliers (name, series, stats, strong=True)
        
                 
        # Check for unbalanced
            
        self.checkForImbalanced (name, stats, series, isObject=False)
      
        self.checkForSkewed (name, stats, series)
        
        self.checkForSparse (name, stats, series)
        
        if stats['unique'] == stats['count']:
            # this is an ID field of uniqie values, needs to be dropped
            self.addRecommendation(name,CLEANDATA_DROP_ID_FIELD)
            self.colsToDrop.append(name)
        

        return
  
    

        
    def reviewDatetime (self, name, stats, series):

        return 

    def reviewBool (self, name, stats, series):

        return 


    # with the total_series_size, 
    #          number of uniqie values
    #          and the precent of uniqie values
    def checkForImbalanced (self, name, stats, series, isObject=False):
    
        
        uniqueValues = stats['unique']
        topPrecent = len(series) / stats['freq']
        topValue = stats['top']
        
        isImbalanced = False
        # test for unbalanced
                   
        if uniqueValues==2 and topPrecent> .80:
            isImbalanced = True
                   
        if uniqueValues==3 and topPrecent> .70:
            isImbalanced = True
            
        if uniqueValues==4 and topPrecent> .60:
            isImbalanced = True
    
        if uniqueValues>=5 and topPrecent> .50:
            isImbalanced = True

           
        if isImbalanced:
            self.addRecommendation(name,CLEANDATA_UNBALANCED_DATA,'Top value {} with {}%'.format(topValue, topPrecent*100.))
        
    def checkForSkewed (self, name, stats, series):
        if hasattr(series, 'cat'):
            return
        skew = series.skew()
        if skew > 4.0:
            self.addRecommendation(name,CLEANDATA_SKEWED_DATA,'Data is skewed with a score of  {}'.format(skew))
        return
    
    
    def checkforOutliers (self, name, series, stats, strong=True):
        
        numItems = len(series)
        
        if stats['unique'] < (numItems * .50) or stats['std']==0:
            return
        
        
        if series.isnull().values.any():
            return
        
        p25 = stats['25%']
        p75 = stats['75%']
    
        p1 = stats['1%']
        p99 = stats['99%']
         
        
        level1 = doubleMADsfromMedian(series)
        if len(level1)>0:
            valuesL1 = [ series[key] for key, value in level1.items() if value==True ]
            
            below25 = [ x for x in valuesL1 if x < p25]
            above75 = [ x for x in valuesL1 if x > p75]
            
            if len(below25) > 0 and strong:
                level2 = doubleMADsfromMedian(below25)
                if len(level2)>0:
                    valuesL2 = [ value for mask, value in zip(np.nditer(level2), below25) if mask==True ]
                    below1 = [ x for x in valuesL2 if x < p1]
                    if len(below1) > 0:
                        self.addRecommendation(name,CLEANDATA_REMOVE_ITEMS_BELOW,np.max(below1))
                
            if len(above75) > 0 and strong:
                level2 = doubleMADsfromMedian(above75)
                if len(level2)>0:
                    valuesL2 = [ value for mask, value in zip(np.nditer(level2), above75) if mask==True ]
                    above99 = [ x for x in valuesL2 if x > p99]
                    if len(above99) > 0:
                        self.addRecommendation(name,CLEANDATA_REMOVE_ITEMS_ABOVE, np.min(above99))
    
        return None
    
        



    def plotExploreHeatMap(self):
               
        # Make the figsize 9 x 8
        plt.figure(figsize=(14,10))

        # Plot heatmap of correlations
        sns.heatmap(self.heatMap, annot=False, cbar=True, cmap='Reds', fmt='.0f')
        plt.show()


   

    def plotNumericColumn(self, name):
        if name in self.dataFrame:
        # Violin plot using the Seaborn library
            sns.violinplot(self.dataFrame[name])
            plt.show()
        
        

    def plotHistogramsAll(self, size=14):
        # Plot histogram grid
        self.dataFrame.hist(figsize=(size,size), xrot=-45)
        # Clear the text "residue"
        plt.show()
        
        
    def plotSegmentation(self, x, y=None, hue=None):
        if x in self.dataFrame:
            if y==None or y in self.dataFrame:
                sns.boxplot(x=x, y=y, data=self.dataFrame)
                plt.show()
                sns.violinplot(x=x, y=y, data=self.dataFrame)
                plt.show()
                # Scatterplot of satisfaction vs. last_evaluation
                if hue in self.dataFrame:
                    sns.lmplot(x=x, y=y, hue=hue, data=self.dataFrame, fit_reg=False)
                    plt.show()

    def plotColumnImportance(self):
        cor = []
        imp = []
        for name in self.topColumns:
            cor.append(self.correlations[self.target][name])
            imp.append(self.featureImportance[name])
    
   
        df = pd.DataFrame({'Correlation to Target': cor,
                          'Feature Importance': imp}, index=self.topColumns)
        df.plot.barh()
        plt.show()



    def plotCorrelations(self):
      
        mask = np.zeros_like(self.correlations, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Make the figsize 9 x 8
        plt.figure(figsize=(14,10))
        # Plot heatmap of correlations
        sns.heatmap(self.correlations, annot=True, mask=mask, cbar=True, cmap='Greens', fmt='.3f')
        plt.show()


    def plotFeatureImportance(self):
        
        col = []
        imp = []
        for c,i in sorted(self.featureImportance.items(), key=lambda item: item[1]):
            col.append(c)
            imp.append(i)
            
        
        # Plot the feature importances of the forest
        
        #names = list(importance.keys())
        #values = list(importance.values())

        #tick_label does the some work as plt.xticks()
        plt.xlabel('Importance')
        plt.ylabel('Column')
        plt.title('Feature Importance')
        #plt.barh(*zip(*sorted(importance.items())))
        plt.barh(col, imp)
        
#        plt.bar(range(len(importance)),values,tick_label=names)
        #plt.savefig('bar.png')
        plt.show()
        return

    def plotCategoryColumn(self, name, figsize=(6,7)):
        if name in self.dataFrame:
            plt.figure(figsize=figsize)
            sns.countplot(y=name, data=self.dataFrame)
            plt.show()        






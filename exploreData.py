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
CLEANDATA_FIX_CASE = 'Fix Case'
CLEANDATA_ZERO_FILL = 'Zero Fill'
CLEANDATA_REMOVE_ITEMS_BELOW = 'Remove Items <='
CLEANDATA_REMOVE_ITEMS_ABOVE = 'Remove Items >='
CLEANDATA_NEW_FEATURE = 'New Feature'
CLEANDATA_DROP_COLUMN = 'Drop Column'
CLEANDATA_NEW_INDICATOR_VARIABLE = 'New Indicator Variable'
CLEANDATA_REMOVE_ITEMS_EQUAL = 'Remove Items Equal'
CLEANDATA_KEEP_ITEMS_EQUAL = 'Keep Items Equal'
CLEANDATA_NUMERIC_MARK_MISSING = 'Numeric Mark Missing'
CLEANDATA_DROP_DUPLICATES = 'Drop Duplicates'
CLEANDATA_CONVERT_DATATYPE = 'Convert Datatype'
CLEANDATA_CONVERT_TO_BOOLEAN = 'Convert Datatype to Boolean'
CLEANDATA_GROUPBY_ROLLUP = 'Group-by Rollup'
CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP =  'Intermediary Group-by Rollup'
CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP = 'Rollup Intermediary Group-by Rollup'
CLEANDATA_JOIN_ROLLUPS = 'Join Group-by Rollups'
CLEANDATA_DIMENSIONALITY_REDUCTION = 'Explore Dimensionality Reduction'
CLEANDATA_PCA_DIMENSIONALITY_REDUCTION = 'PCA Dimensionality Reduction'
CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION =  'Threshold Dimensionality Reduction'
CLEANDATA_JOIN_TABLES = 'Join tables'
CLEANDATA_SET_BATCH_TABLES = 'Set table to run in batch'
CLEANDATA_DROP_NA = 'Drop rows with missing values'
CLEANDATA_DROP_NA_FOR_COLUMN = 'Drop rows with missing values in column'
CLEANDATA_SET_CATEGORY_DATATYPE = 'Convert object type to category'
CLEANDATA_NEW_FEATURE_DATE_DIFFERENCE_MONTHS = 'New feature date difference in months'

exploreAutoFunctions = [
                        CLEANDATA_DROP_DUPLICATES
                        ]



exploreManualFunctions = [CLEANDATA_DROP_DUPLICATES,
                          CLEANDATA_MARK_MISSING, 
                          CLEANDATA_FIX_CASE,
                          CLEANDATA_ZERO_FILL,
                          CLEANDATA_REBUCKET ,
                          CLEANDATA_REMOVE_ITEMS_BELOW,
                          CLEANDATA_REMOVE_ITEMS_ABOVE,
                          CLEANDATA_NEW_FEATURE,
                          CLEANDATA_NEW_FEATURE_DATE_DIFFERENCE_MONTHS,
                          CLEANDATA_DROP_COLUMN,
                          CLEANDATA_NEW_INDICATOR_VARIABLE,
                          CLEANDATA_REMOVE_ITEMS_EQUAL,
                          CLEANDATA_KEEP_ITEMS_EQUAL,
                          CLEANDATA_NUMERIC_MARK_MISSING,
                          CLEANDATA_CONVERT_DATATYPE,
                          CLEANDATA_CONVERT_TO_BOOLEAN,
                          CLEANDATA_GROUPBY_ROLLUP ,
                          CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP,
                          CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP,
                          CLEANDATA_JOIN_ROLLUPS,
                          CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION,
                          CLEANDATA_PCA_DIMENSIONALITY_REDUCTION,
                          CLEANDATA_JOIN_TABLES,
                          CLEANDATA_SET_BATCH_TABLES,
                          CLEANDATA_DROP_NA,
                          CLEANDATA_DROP_NA_FOR_COLUMN,
                          CLEANDATA_SET_CATEGORY_DATATYPE
                          ]


exploreHeatMap = [CLEANDATA_MARK_MISSING, 
                  CLEANDATA_FIX_CASE,
                  CLEANDATA_ZERO_FILL,
                  CLEANDATA_DROP_DUPLICATES,
                  CLEANDATA_REBUCKET ,
                  CLEANDATA_REMOVE_ITEMS_BELOW,
                  CLEANDATA_REMOVE_ITEMS_ABOVE,
                  CLEANDATA_NEW_FEATURE,
                  CLEANDATA_DROP_COLUMN,
                  CLEANDATA_NEW_INDICATOR_VARIABLE,
                  CLEANDATA_REMOVE_ITEMS_EQUAL,
                  CLEANDATA_NUMERIC_MARK_MISSING,
                  CLEANDATA_CONVERT_DATATYPE,
                  CLEANDATA_GROUPBY_ROLLUP ,
                  CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP,
                  CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP,
                  CLEANDATA_JOIN_ROLLUPS,
                  CLEANDATA_DIMENSIONALITY_REDUCTION
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
    stats['dataType'] = series.dtype
    #print ('dtype=', series.dtype)
    
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
   
    
 
    # Status for Objects
    if series.dtype == 'object':
        stats['unique'] = s['unique']
        stats['top'] = s['top']
        stats['freq'] = s['freq']
        stats['histogram'] = getHistogram(series)

    elif hasattr(series,'cat'):
        stats['unique'] = s['unique']
        stats['top'] = s['top']
        stats['freq'] = s['freq']
        stats['histogram'] = getHistogram(series)
        
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
         
   
    # test for Binary
    if stats['unique'] == 2 and series.dtype != 'object':
        stats['binary']=True
    else:
        stats['binary']=False

    return stats


class exploreData(object):
    
    def __init__(self, df, project):
        
       
        self.columns = {}
        self.recommendations = {}
        self.dataFrame = df
        
        # Init explore heatmap to zeros
        self.heatMap = pd.DataFrame(np.zeros([len(exploreHeatMap),len(df.columns)], dtype=float), index=exploreHeatMap, columns=df.columns)

        
        d  = np.zeros([len(exploreHeatMap),len(df.columns)], dtype=float)
        self.headmap = pd.DataFrame(d, index=exploreHeatMap, columns=df.columns)

        
        # Loop through each columne and analize:


        # Loop through categorical feature names and print each one
        #for x in df.dtypes[(df.dtypes=='object')].index:
        #    print (x)   
        for name in df.dtypes.index:
        
            type = df.dtypes[name]
            #print (name, type)
            stats = getStats(df[name])
            self.columns[name] = stats
            
            #print ('\nStatus for=',name)
            #print (self.columns[name])
            
            if type == 'object':
                self.reviewObjects(name, stats, df[name], project)
            elif hasattr(df[name], 'cat'):
                self.reviewObjects(name, stats, df[name], project)
            elif type == 'datetime64[ns]':
                self.reviewDatetime(name, stats, df[name])
            elif type == 'bool':
                self.reviewBool(name, stats, df[name])
 
            else:
                self.reviewNumbers(name, stats, df[name])   
            
        return
#        try:

        
#        except AssertionError:
#            raise Exception(self.KAT_TABLE_ERROR)
            
#        except Exception as e:
#            raise e
  
    def __str__ (self):
        str = ''
        for key, value in self.recommendations.items():
            str += '\nRecommendations for column {}\n'.format(key)
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
            str += '   max:    {:<12}   uniqie: {:<12}\n\n'.format(dStats['max'],dStats['unique'] )
#            str += '    1%:    {:<10}   2.5%:   {:<10}   5%:   {:<10}      25%:   {:<10}    50%:   {:<10}\n'.format(stats['1%'], stats['2.5%'], stats['5%'] , stats['25%'] ,stats['50%'])
#            str += '   50%:    {:<10}   75%:    {:<10}  95%:   {:<10}    97.5%:   {:<10}    99%:   {:<10}\n'.format(stats['50%'], stats['75%'], stats['95%'] , stats['97.5%'] , stats['99%'])
            #print (name)
            if stats['dataType'] == 'bool':
                pass
            elif stats['dataType']=='datetime64':
                pass
            elif hasattr(stats['dataType'], 'cat'):
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
        print ('\n\nData Summary:')
        print ('\n\nFirst {} rows: '.format(rows))
        print  (self.dataFrame.head(n=rows))
        print  ('\n\nLast {} rows: '.format(rows))
        print  (self.dataFrame.tail(n=rows))
        print  ('\n')
        


    def plotCategoryColumn(self, name, figsize=(6,7)):
        if name in self.dataFrame:
            plt.figure(figsize=figsize)
            sns.countplot(y=name, data=self.dataFrame)
            plt.show()        


    def getColumnValues(self, name):
        if name in self.dataFrame:
            return self.dataFrame[name].unique().tolist()
        return []

        
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


    def plotCorrelations(self):


        correlations = self.dataFrame.corr() * 100
        
        mask = np.zeros_like(correlations, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        

        # Make the figsize 9 x 8
        plt.figure(figsize=(14,10))


        # Plot heatmap of correlations
        sns.heatmap(correlations, annot=True, mask=mask, cbar=True, cmap='Greens', fmt='.0f')
        plt.show()




    def plotExploreHeatMap(self):
               
        # Make the figsize 9 x 8
        plt.figure(figsize=(14,10))


        # Plot heatmap of correlations
        sns.heatmap(self.heatMap, annot=False, cbar=True, cmap='Reds', fmt='.0f')
        plt.show()


    def reviewObjects (self, name, stats, data, project):
       
        
        if stats['nullCount'] > 0:
            self.addRecommendation(name,CLEANDATA_MARK_MISSING, stats['nullCount'])
    
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
                #print ("item, lower = ",item, lower)
                if item != lower:
                    if lower in histogram:
                        #print ('   --> add recommendations',lower, item)
                        self.addRecommendation(name,CLEANDATA_FIX_CASE,[lower, item], lower+CLEANDATA_BREAK+str(item))
                upper = item.upper()
                if item != upper:
                    if upper in histogram:
                        self.addRecommendation(name,CLEANDATA_FIX_CASE+CLEANDATA_BREAK, [upper, item], upper+CLEANDATA_BREAK+str(item))
    
    
        
        return 
    
    
    
    def reviewNumbers (self, name, stats, series):
        
        
        if stats['nullCount'] > 0:
            if stats['binary']==True:
                self.addRecommendation(name,CLEANDATA_ZERO_FILL,name)
            else:
                self.addRecommendation(name,CLEANDATA_NUMERIC_MARK_MISSING,name)
        
        #recommendations['Remove the top 1% for '+ name] =  stats['99%']
        #recommendations['Remove the bottom 1% for '+ name] =  stats['1%']
        
        self.checkforOutliers (name, series, stats, strong=True)
        
      
    
    
        
    def reviewDatetime (self, name, stats, series):

        return 

    def reviewBool (self, name, stats, series):

        return 
   
    
    def checkforOutliers (self, name, series, stats, strong=True):
        
        if stats['unique'] < 10 or stats['std']==0:
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
                level2 = doubleMADsfromMedian(below25, 7)
                if len(level2)>0:
                    valuesL2 = [ value for mask, value in zip(np.nditer(level2), below25) if mask==True ]
                    below1 = [ x for x in valuesL2 if x < p1]
                    if len(below1) > 0:
                        self.addRecommendation(name,CLEANDATA_REMOVE_ITEMS_BELOW,np.max(below1))
                
            if len(above75) > 0 and strong:
                level2 = doubleMADsfromMedian(above75, 7)
                if len(level2)>0:
                    valuesL2 = [ value for mask, value in zip(np.nditer(level2), above75) if mask==True ]
                    above99 = [ x for x in valuesL2 if x > p99]
                    if len(above99) > 0:
                        self.addRecommendation(name,CLEANDATA_REMOVE_ITEMS_ABOVE, np.min(above99))
    
        return None
    
    
    def addRecommendation (self, name, explore, value, key=None):
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
           
                
                    



## Filter and display only df.dtypes that are 'object'
#print (df.dtypes[(df.dtypes=='object')])
#
#
#
## Plot histogram grid
#df.hist(figsize=(14,14), xrot=-45)
## Clear the text "residue"
#plt.show() 
#
#
## Summarize numerical features
#df['married'].describe()
#
#
## Summarize categorical features
#df.describe(include=['object'])
#
#
## Bar plot for 'exterior_walls'
#sns.countplot(y='exterior_walls', data=df)
#
## Plot bar plot for each categorical feature
#for x in df.dtypes[(df.dtypes=='object')].index:
#    sns.countplot(y=x, data=df)
#    plt.show()
#    
#    
#    
## Segment tx_price by property_type and plot distributions
#sns.boxplot(y='property_type', x='tx_price', data=df)
#
#
## Segment by property_type and display the means within each class
#df.groupby('property_type').mean()
#
#
## Segment sqft by sqft and property_type distributions
#sns.boxplot(y='property_type', x='sqft', data=df)
#
#
## Segment by property_type and display the means and standard deviations within each class
#df.groupby('property_type').describe()
#
#
## Calculate correlations between numeric features
#correlations = df.corr()
#print (correlations)
#
## Generate a mask for the upper triangle
#mask = np.zeros_like(correlations, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#
#
#c2 = df.corr() * 100
#
## Make the figsize 9 x 8
#plt.figure(figsize=(14,10))
#
#
#
## Plot heatmap of correlations
#sns.heatmap(c2, annot=True, mask=mask, cbar=True, cmap='Greens', fmt='.0f')

        
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



from .exploreData import * 
import mlLib.utility as utility


#import exploreData as ed
#import utility 

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA






######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanMarkMissing(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_MARK_MISSING
        return
  
    def execute(self, df, cleaningRules=None):       
        df[self.name].fillna('Missing', inplace = True)
        return None
 
######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######    
class cleanNumericMarkMissing(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_NUMERIC_MARK_MISSING
        return
  
    def execute(self, df, cleaningRules=None):
        df[self.name + '_missing'] = (df[self.name].isnull()).astype(int)
        df[self.name] = df[self.name].fillna(0)
        return None
 
######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanRebucket(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_REBUCKET
        return
  
    def execute(self, df, cleaningRules=None):
        df[self.name].replace(self.value[0], self.value[1], inplace=True)
        return None
 
 ######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanFixCase(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_FIX_CASE
        return
    
    def execute(self, df, cleaningRules=None):
        df[self.name].replace(self.value[0], self.value[1], inplace=True)
        return None


######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanZeroFill(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_ZERO_FILL
        return
   
    def execute(self, df, cleaningRules=None):
        df[self.name].fillna(0, inplace = True)
        return None


######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanRemoveItemsBelow(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_REMOVE_ITEMS_BELOW
        return
   
    def execute(self, df,cleaningRules=None):
        df.drop(df[df[self.name] <= self.value].index, inplace=True)
        return None


######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanRemoveItemsAbove(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_REMOVE_ITEMS_ABOVE
        return
 
    
    def execute(self, df, cleaningRules=None):
        df.drop(df[df[self.name] >= self.value].index, inplace=True)
        return None


######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanRemoveItemsEqual(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_REMOVE_ITEMS_EQUAL
        return
   
    def execute(self, df, cleaningRules=None):
        if self.value:
            df.drop(df[df[self.name] == self.value].index, inplace=True)
        else:
            df.drop(df[df[self.name].isnull()].index, inplace=True)
        return None



class cleanKeepItemsEqual(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_KEEP_ITEMS_EQUAL
        return
   
    def execute(self, df, cleaningRules=None):
        if type(self.value) is not list:
            keepList = [self.value]
        else:
            keepList = self.value
            
            
        if self.name in df:
            valList = df[self.name].unique().tolist()
        else:
            utility.errorLog('Unable to Keep list for {} and values {}'.format(self.name, self.value))
            return None
            
        for toDrop in valList:
            if toDrop in keepList:
                pass
            else:
                df.drop(df[df[self.name] == toDrop].index, inplace=True)
        return None


######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanConvertDatatype(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_CONVERT_DATATYPE
        return
   
    def execute(self, df, cleaningRules=None):
        df[self.name] = df[self.name].astype(self.value)
        return None



class cleanConvertToBoolean(object):
    def __init__ (self, name, value=None):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_CONVERT_TO_BOOLEAN
        return
   
    def execute(self, df, cleaningRules=None):
        if self.value is None:  
            trueValue = True
        else:
            trueValue = self.value
    
        df[self.name] =  (df[self.name]==trueValue).astype(int)
        return None



class cleanSetCatgory(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_SET_CATEGORY_DATATYPE
        return
   
    def execute(self, df, cleaningRules=None):
        df[self.name] = df[self.name].astype('category',categories=self.value)
        return None


######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanDropColumn(object):
    def __init__ (self, name, value=None):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_DROP_COLUMN
        return
 
    
    def execute(self, df, cleaningRules=None):
        if type(self.name) is not list:
            if self.name in df:
                df.drop(self.name, axis=1, inplace=True)
        else:
            for name in self.name:
                if name in df:
                    df.drop(name, axis=1, inplace=True)
                else:
                    utility.runLog( 'Error: Columns {} not found to drop'.format(name))
        return None
 
######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanNewFeature(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_NEW_FEATURE
        return
 
    def execute(self, df, cleaningRules=None):
        eq = equationPrep(df, self.name, self.value, indicator=False)  
        exec(eq, {'__builtins__' : None }, {'df' : df })
        return None



class cleanNewFeatureDateDifferenceMonths(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_NEW_FEATURE_DATE_DIFFERENCE_MONTHS
        return
 
    def execute(self, df, cleaningRules=None):
        df[self.name] = ((abs(df[self.value[0]]- df[self.value[1]]))/np.timedelta64(1, 'M'))
        df[self.name] = df[self.name].astype(int)
        return None
    

######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanNewIndicatorVariable(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_NEW_INDICATOR_VARIABLE
        return
 
    def execute(self, df, cleaningRules=None):
        eq = equationPrep(df, self.name, self.value, indicator=True)  
        exec(eq, {'__builtins__' : None }, {'df' : df , 'int' : int })
        return None


######
#
# project.addManualRuleForDefault(CLEANDATA_GROUPBY_ROLLUP, ['CustomerID','Sales'], { 'total_sales' : 'sum', 
#                                                  'avg_product_value' : 'mean' })
#           name = [0]=list of of column names to group by from the intermediate table
#                  [1]= the column to aggegrate
#           value = dictoinary of new column names and function to agg (values are 'nunique', 'count', 'sum', 'mean', 'min', 'max' )
#
#
######
class cleanDropDuplicates(object):
    def __init__ (self, name=None, value=None):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_DROP_DUPLICATES
        
    def execute(self, df, cleaningRules=None):
        # drop duplicates
        df.drop_duplicates(inplace=True)
        return None



"""

Re- Shape Data cleaning rules

"""


######
#
# Description: create a rollup
#
# example:
#       project.addManualRuleForDefault(CLEANDATA_GROUPBY_ROLLUP, ['CustomerID','Sales'], { 'total_sales' : 'sum', 
#                                                  'avg_product_value' : 'mean' })
# params
#           name = [0]=list of of column names to group by from the intermediate table
#                  [1]= the column to aggegrate
#           value = dictoinary of new column names and function to agg (values are 'nunique', 'count', 'sum', 'mean', 'min', 'max' )
#
#
######
class cleanRollup(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_GROUPBY_ROLLUP
        
    def execute(self, df, cleaningRules=None):        
        # Roll up  data
        # get the agg functions (mean, max, etx)
        toAgg = [val for key, val in self.value.items()]
        
        # run the group by
        newCol = df.groupby(self.name[0])[self.name[1]].agg( toAgg )
    
        # rename the columns from the defaults
        toRename = {val: key for key, val in self.value.items()}
        newCol.rename( columns=toRename, inplace=True)
        
        # save the new rollup in the rollup list
        cleaningRules.rollups.append(newCol)
        return None
      



######
#
# Description: Create an intermediate group as rollup to be further rolluped
#
# example:
#    project.addManualRuleForDefault(CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP, [['CustomerID','InvoiceNo'],'Sales'], { 'cart_value' : 'sum' } )
#                                                                                                  'max_cart_value' : 'max'})
#    args:
#           name = [0]=list of of column names to group by from the intermediate table
#                  [1]= the column to aggegrate
#           value = dictoinary of new column names and function to agg (values are 'nunique', 'count', 'sum', 'mean', 'min', 'max' )
#
######
class cleanIntermediaryGroup(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP
        
    def execute(self, df, cleaningRules=None):


        # get the agg functions (mean, max, etx)
        toAgg = [val for key, val in self.value.items()]
        
        # run the group by
        cleaningRules.intermediary = df.groupby(self.name[0])[self.name[1]].agg( toAgg )
    
        # rename the columns from the defaults
        toRename = {val: key for key, val in self.value.items()}
        cleaningRules.intermediary.rename( columns=toRename, inplace=True)
        
        return None




######
#
# Description: take the intermediate table and create a rollup
        
# Example:
#       project.addManualRuleForDefault(CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP, ['CustomerID','cart_value'], { 'avg_cart_value' : 'mean',
#                                                                                                    'min_cart_value' : 'min',
#                                                                                                    'max_cart_value' : 'max'})
#    args:
#           name = [0]=list of of column names to group by from the intermediate table
#                  [1]= the column to aggegrate
#           value = dictoinary of new column names and function to agg (values are 'nunique', 'count', 'sum', 'mean', 'min', 'max' )
#
######
class cleanRollupIntermediary(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP
        
    def execute(self, df, cleaningRules=None):
              
        # get the agg functions (mean, max, etx)
        toAgg = [val for key, val in self.value.items()]
        
        # run the group by
        newCol = cleaningRules.intermediary.groupby(self.name[0])[self.name[1]].agg( toAgg )
    
        # rename the columns from the defaults
        toRename = {val: key for key, val in self.value.items()}
        newCol.rename( columns=toRename, inplace=True)
        
        # save the new rollup in the rollup list
        cleaningRules.rollups.append(newCol)

        self.intermediary = None

        return None

 

######
#
# Description: create a dataframe from the rollups
# example:
#           project.addManualRuleForDefault(CLEANDATA_JOIN_ROLLUPS,'customer_df',None)
#   args:
#       Values: name = new 'table' name
#
######
class cleanJoinRollup(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_JOIN_ROLLUPS
        
    def execute(self, df, cleaningRules=None):
        if len(cleaningRules.rollups) > 1:
            main = cleaningRules.rollups.pop()
            cleaningRules.project.preppedTablesDF[self.name] = main.join(cleaningRules.rollups)
        elif len(cleaningRules.rollup)==1:
            cleaningRules.project.preppedTablesDF[self.name] = self.rollups.pop()
        else:
            utility.raiseError('No Rollups to Join')
            
        self.rollups = []
   
        return None


######
#
# Description: 
# example:
#           project.addManualRuleForDefault(CLEANDATA_PCA_DIMENSIONALITY_REDUCTION,[ColumnName, groupByKey], value)
#   args:
#       name = [columnName (to be evaluated), GroupByNnme]
#       Values: Name of the column  
#
######
class cleanPCADimensionalityReduction(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_PCA_DIMENSIONALITY_REDUCTION
        
    def execute(self, df, cleaningRules=None):
        cleaningRules.project.preppedTablesDF[self.value] = pca(df, self.name[0], self.name[1], cleaningRules.project)
        return None

######
#
# Description: 
# example:
#           project.addManualRuleForDefault(CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION,[ColumnName, groupByKey], Value)
#   args:
#       name = [columnName (to be evaluated), GroupByNnme]
#       Values: Name of the new table
#       
#
######
class cleanThresholdDimensionalityReduction(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION
        
    def execute(self, df, cleaningRules=None):
        cleaningRules.project.preppedTablesDF[self.value] =  threshold(df, self.name[0], self.name[1], cleaningRules.project)
        return None



######
#
# Description: 
# example:
#          project.addManualRuleForDefault(CLEANDATA_JOIN_TABLES,['base','pca_item_data'],'pca_df')

#   args:
#       Name: List of tables to join
#       Values: name of new tables
        
#
######
class cleanJoinTables(object):
    def __init__ (self, name, value):
        self.name = name 
        self.value = value
        self.ruleName = CLEANDATA_JOIN_TABLES
        
    def execute(self, df, cleaningRules=None):

        # test for file names
        for x in self.name:
            if x not in cleaningRules.project.preppedTablesDF:
                utility.raiseError('Table "{}" not found'.format(x))
                return None
        if len(self.name) < 2:
            utility.raiseError('Not enough tables to join. sent='.format(selfname))
            return None

        main = None
        lst = []
        for tableName in self.name:
            if main is None:
                main = cleaningRules.project.preppedTablesDF[tableName]
            else:
                lst.append(cleaningRules.project.preppedTablesDF[tableName])

        cleaningRules.project.preppedTablesDF[self.value] = main.join(lst)



######
#
# Description: 
# example:
#          project.addManualRuleForDefault(CLEANDATA_SET_BATCH_TABLES,['customer_df', 'threshold_df', 'pca_df'], None)

#   args:
#       Values: Name of the column
        
#
######
class cleanSetBatchTables(object):
    def __init__ (self, name, value):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_SET_BATCH_TABLES
        
    def execute(self, df, cleaningRules=None):
        for n in self.name:
            if n not in cleaningRules.project.preppedTablesDF:
                utility.raiseError('Table not found: '+n)
                return None
        cleaningRules.project.batchTablesList = self.name
        return None


######
#
# Description: 
#
# example:
#       
# params
#           name = 
#           value = 
#
#
######
class cleanDropNA(object):
    def __init__ (self, name=None, value=None):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_DROP_NA
        return
   
    def execute(self, df, cleaningRules=None):
        df.dropna(inplace=True)
        return None


class cleanDropNAForColumn(object):
    def __init__ (self, name, value=None):
        self.name = name
        self.value = value
        self.ruleName = CLEANDATA_DROP_NA_FOR_COLUMN
        return
   
    def execute(self, df, cleaningRules=None):
        df[self.name].dropna(inplace=True)
        return None



def threshold(df, columnName, indexKeyName, project):
    # Create data by aggregating at customer level
    dummies = pd.get_dummies( df[columnName] )

    # Add CustomerID to toy_item_dummies
    dummies[indexKeyName] = df[indexKeyName]
    data = dummies.groupby(indexKeyName).sum()
    
    top_items = data.sum().sort_values().tail(project.clusterDimensionThreshold).index
    
    return data[top_items]
    



def pca(df, columnName, indexKeyName, project):
    # Create data by aggregating at customer level
    dummies = pd.get_dummies( df[columnName] )

    # Add CustomerID to toy_item_dummies
    dummies[indexKeyName] = df[indexKeyName]
    data = dummies.groupby(indexKeyName).sum()
    
    # Initialize instance of StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform item_data
    data_scaled = scaler.fit_transform(data)    

    # Initialize and fit a PCA transformation
    pca = PCA()
    
    # Fit the instance
    pca.fit(data_scaled)
    
    # Generate new features
    PC_features = pca.transform(data_scaled)
    
    # Cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
       
    # Find the number of components
    threshold = len(cumulative_explained_variance) - 1
    for i in range(len(cumulative_explained_variance)):
        if cumulative_explained_variance[i] > project.varianceThreshold:
            threshold = i
            break
             
    # Initialize PCA transformation, only keeping 125 components
    pca = PCA(n_components=threshold)
    
    # Fit and transform item_data_scaled
    PC_features = pca.fit_transform(data_scaled)
    
    # Put PC_items into a dataframe
    pca_df = pd.DataFrame(PC_features)
    
    # Name the columns
    pca_df.columns = ['PC{}'.format(i + 1) for i in range(PC_features.shape[1])]
    
    # Update its index
    pca_df.index = data.index
    
    return pca_df

    
def equationPrep(df, name, value, indicator):
    
    def fixSpacing(s):
        opChar = ['&','+','-','/','*','(',')','=','>','<','|','~','^']
        str = ''
        lastWasChar = False
        lastWasOp = False
        for x in s:
            if x in opChar:
                lastWasOp = True
                if lastWasChar:
                    str += ' '
                    lastWasChar = False
            else:
                lastWasChar = True
                if lastWasOp:
                    str += ' '
                    lastWasOp = False
                
            str += x          
        return str

    
    fixedSpaces = fixSpacing(value)
    words = fixedSpaces.split(' ')

    eq = ''
    for w in words:
        if w in df:
            eq += "df['{}'] ".format(w)
        elif indicator and w.lower() == 'and':
            eq += ' & '
        elif indicator and w.lower() == 'or':
            eq += ' | '
        else:
            eq += w + ' '
          
    if indicator:
        return "df['{}'] = ({}).astype(int)".format(name, eq)
    else:
        return "df['{}'] = {}".format(name, eq)
        



def cleanData(df, cleaningRules):
        
    # Execute rules
    if cleaningRules is not None:
        for rule in cleaningRules.rules:
            if type(rule.name) is not list:
                if rule.name in df.columns:
                    utility.runLog( 'Running Rule {} for Column {} with value {}'.format(rule.ruleName, rule.name, rule.value))
                elif rule.name is not None:
                    if rule.value is None:
                        utility.runLog( 'Running Rule {} '.format(rule.ruleName))
                    else:
                        utility.runLog( 'Running Rule {} for {} with value {}'.format(rule.ruleName, rule.name, rule.value))
                else:
                    utility.runLog( 'Running Rule {} '.format(rule.ruleName))
            else:
                if rule.value is None:
                    utility.runLog( 'Running Rule {} for {}'.format(rule.ruleName,rule.name))
                else:
                    utility.runLog( 'Running Rule {} for {} with value {}'.format(rule.ruleName, rule.name, rule.value)) 
            #sizeBefore = df.shape    
            rule.execute(df, cleaningRules)   
            #print ('Size Before Rule {} and After {}'.format(sizeBefore, df.shape ))
    return 
    
    
     
class cleaningRules(object):

    def __init__ (self, project, explore):
        
        if project.dropDuplicates:
            self.rules = [cleanDropDuplicates()]
        else:
            self.rules = []
        for columnName, paramater in explore.recommendations.items():          
            for functionName, val in paramater.items():
                # Pull out the name in the function name to get the function
                i = functionName.find(CLEANDATA_BREAK)
                if i == -1:
                    functionNameSub = functionName
                else:
                    functionNameSub = functionName[:i]
                if functionNameSub in exploreAutoFunctions:
                    if functionNameSub == CLEANDATA_MARK_MISSING:
                        cleanFunction = cleanMarkMissing(columnName, val)
                    elif functionNameSub == CLEANDATA_FIX_CASE:
                        #print ("fix case", functionName, len(CLEANDATA_FIX_CASE))
                        cleanFunction =  cleanFixCase(columnName, val)
                    elif functionNameSub == CLEANDATA_ZERO_FILL:
                        cleanFunction =  cleanZeroFill(columnName, val)                                        
                    self.rules.append(cleanFunction)
        self.rollups = []
        self.intermediary = None
        self.project = project
        return

       
    def addManualRule(self, functionName, columnName, value , df=None):
        if functionName in exploreManualFunctions:
            if functionName == CLEANDATA_MARK_MISSING:
                cleanFunction = cleanMarkMissing(columnName, value)
            elif functionName == CLEANDATA_FIX_CASE:
                cleanFunction =  cleanFixCase(columnName, value)
            elif functionName == CLEANDATA_ZERO_FILL:
                cleanFunction =  cleanZeroFill(columnName, value)
            elif functionName == CLEANDATA_REBUCKET:
                cleanFunction = cleanRebucket(columnName, value)
            elif functionName == CLEANDATA_REMOVE_ITEMS_BELOW:
                cleanFunction = cleanRemoveItemsBelow(columnName, value)
            elif functionName == CLEANDATA_REMOVE_ITEMS_ABOVE:
                cleanFunction = cleanRemoveItemsAbove(columnName, value)
            elif functionName == CLEANDATA_REMOVE_ITEMS_EQUAL:
                cleanFunction = cleanRemoveItemsEqual(columnName, value)
            elif functionName == CLEANDATA_KEEP_ITEMS_EQUAL:
                cleanFunction = cleanKeepItemsEqual(columnName, value)
            elif functionName == CLEANDATA_NEW_FEATURE:
                cleanFunction = cleanNewFeature(columnName, value)
            elif functionName == CLEANDATA_NEW_FEATURE_DATE_DIFFERENCE_MONTHS:
                cleanFunction = cleanNewFeatureDateDifferenceMonths(columnName, value)
            elif functionName == CLEANDATA_DROP_COLUMN:
                cleanFunction = cleanDropColumn(columnName, value)
            elif functionName == CLEANDATA_NEW_INDICATOR_VARIABLE:
                cleanFunction = cleanNewIndicatorVariable(columnName, value)
            elif functionName == CLEANDATA_NUMERIC_MARK_MISSING:
                cleanFunction = cleanNumericMarkMissing(columnName, value)
            elif functionName == CLEANDATA_CONVERT_DATATYPE:
                cleanFunction = cleanConvertDatatype(columnName, value)
            elif functionName == CLEANDATA_CONVERT_TO_BOOLEAN:
                cleanFunction = cleanConvertToBoolean(columnName, value)
            elif functionName == CLEANDATA_DROP_NA:
                cleanFunction = cleanDropNA()
            elif functionName == CLEANDATA_DROP_NA_FOR_COLUMN:
                cleanFunction = cleanDropNAForColumn(columnName)
            elif functionName == CLEANDATA_SET_CATEGORY_DATATYPE:
                if value is not None:
                    columns = value
                elif columnName in df:
                    tmp = df[columnName].unique()
                    columns = []
                    for x in tmp:
                        if type(x)==str:
                            columns.append(x)                
                else:
                    return 
                cleanFunction = cleanSetCatgory(columnName, columns)

            # Data shaping functions
            elif functionName == CLEANDATA_GROUPBY_ROLLUP:
                cleanFunction = cleanRollup(columnName, value)
            elif functionName == CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP:
                cleanFunction = cleanIntermediaryGroup(columnName, value)
            elif functionName == CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP:
                cleanFunction = cleanRollupIntermediary(columnName, value)
            elif functionName == CLEANDATA_JOIN_ROLLUPS:
                cleanFunction = cleanJoinRollup(columnName, value)


            elif functionName == CLEANDATA_PCA_DIMENSIONALITY_REDUCTION:
                cleanFunction = cleanPCADimensionalityReduction(columnName, value)
            elif functionName == CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION:
                cleanFunction = cleanThresholdDimensionalityReduction(columnName, value)
                
            elif functionName == CLEANDATA_JOIN_TABLES:
                cleanFunction = cleanJoinTables(columnName, value)
            elif functionName == CLEANDATA_SET_BATCH_TABLES:
                cleanFunction = cleanSetBatchTables(columnName, value)
            
            else:
                return



            self.rules.append(cleanFunction)
            return 
            
            
    def addAndRunManualRule(self, df, functionName, columnName, value ):
        rule = self.addManualRule(functionName, columnName, value)
        cleanData(df, [rule])
        
      


  
        
    
#s1 = '((last_evaluation>=0.8)and(satisfaction>=0.7))'
#s2 = '((last_evaluation>0.8)or(satisfaction>=0.7))'
#s3 = '((last_evaluation>0.8)and(satisfaction>0.7))'
#s4 = 'num_schools*median_school'
#s5 = 'tx_year-year_built'
#s6 = '( beds == 2 ) & ( baths == 2 )'
#df = ['num_schools','year_built','median_school','last_evaluation','satisfaction','tx_year', 'beds', 'baths']
#print (equationPrep(df, 'results_col', s1, True))
#print (equationPrep(df, 'results_col', s2, True))
#print (equationPrep(df, 'results_col', s3, True))
#print (equationPrep(df, 'results_col', s4, False))
#print (equationPrep(df, 'results_col', s5, False))
#print (equationPrep(df, 'results_col', s6, False))


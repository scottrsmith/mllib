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
import mlLib.mlUtility as mlUtility


#import exploreData as ed
#import utility 

import numpy as np
import pandas as pd
import math
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.utils.testing import ignore_warnings







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
    def __init__ (self, name, value=None ,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_MARK_MISSING
        return
  
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        if self.value is None:     
            df[self.name].fillna('Missing', inplace = True)
        else:
            df[self.name].fillna(self.value, inplace = True)
        
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
    def __init__ (self, name, value=None,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_NUMERIC_MARK_MISSING
        return
  
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        df[self.name + '_missing'] = (df[self.name].isnull()).astype(int)
        df[self.name] = df[self.name].fillna(0)
        return None
 
######
#
# Description: 
#
# example: project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET, 'department', [['information_technology'], 'IT'])
#       
# params
#           name = 
#           value = 
#
#
######
class cleanRebucket(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REBUCKET
        return
  
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        df[self.name].replace(self.value[0], self.value[1], inplace=True)
        return None


######
#
# Description: 
#
# example: project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET, 'department', [['information_technology'], 'IT'])
#       
# params
#           name = 
#           value = 
#
#
######
class cleanRebucketWhere(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REBUCKET_WHERE
        return
  
    def execute(self, df, project, cleaningRules=None):
        column, op, upper, newValue = tuple(self.value)       
        if op=='less':
            df.loc[ df[column] <= upper, self.name]  = newValue
        elif op=='between':
            lower, upper = tuple(upper)
            df.loc[ (df[column] > lower) & (df[column] <= upper), self.name]  = newValue
        elif op=='more':
            df.loc[ df[column] > upper, self.name]  = newValue        
        return None

######
#
# Description: 
#
# example: project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET_TO_BINARY, 'Cabin', 'col_name')
#       
# params
#           name = 
#           value = new column name
#
#
######
class cleanRebucketBinary(object):
    def __init__ (self, name, value, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REBUCKET_TO_BINARY
        return
  
    def execute(self, df, project, cleaningRules=None):       
        if self.name not in df:
            return None
        df[self.value] = df[self.name].apply(lambda x: 0 if type(x) == float else 1)
        return None

 
 
#####
#
# Description: 
#
# example: project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET_TO_BINARY, 'Cabin', 'col_name')
#       
# params
#           name = 
#           value = new column name
#
#
######
class cleanFixSkew(object):
    def __init__ (self, name, value, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_FIX_SKEW
        return
  
    def execute(self, df, project, cleaningRules=None):       
        if self.name not in df:
            return None
        # Apply log to reduce skewness distribution
        df[self.name] = df[self.name].map(lambda i: np.log(i) if i > 0 else 0)       
        
        return None


#####
#
# Description: 
#
# example: project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET_TO_BINARY, 'Cabin', 'col_name')
#       
# params
#           name = 
#           value = new column name
#
#
######
class cleanStripToAlpha(object):
    def __init__ (self, name, value, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_STRIP_TO_ALPHA
        return
  
    def execute(self, df, project, cleaningRules=None):       
        # Apply log to reduce skewness distribution
        if self.name not in df:
            return None

        newCol = []
        for d in df[self.name].tolist():
            if not d.isdigit() :
                #PERMITTED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" 
                #newString = (''.join(c for c in d if c in PERMITTED_CHARS))
                regex = re.compile('[^a-zA-Z]')
                newString = regex.sub('',d)
                newCol.append(newString.upper())
            else:
                newCol.append(self.value)
                
        df[self.name] = newCol         
        return None



 
######
#
# Description: 
#
# example: project.addManualRuleForDefault(ed.CLEANDATA_REBUCKET_BY_RANGE, 'Fare',  [ (None,'less',7.91,0),
#                                                                  (7.91,'between',14.454,1),
#                                                                  (14.454,'between',31,2),
#                                                                  (None,'more',31.2,3) ])
#
# Less = X <= value
# between = X > Value and X<= Value
# more = X > Value
# params 
#           name = 
#           value = 
#
#
######
class cleanRebucketByRange(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REBUCKET_BY_RANGE
        return
  
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        for rule in self.value:
            lower, op, upper, newValue = tuple(rule)
            if op=='less':
                df.loc[ df[self.name] <= upper, self.name]  = newValue
            elif op=='between':
                df.loc[ (df[self.name] > lower) & (df[self.name] <= upper), self.name]  = newValue
            elif op=='more':
                df.loc[ df[self.name] > upper, self.name]  = newValue        
        return None


######
#
# Description: 
#
# project.addManualRuleForInclude(ed.CLEANDATA_REBUCKET_BY_LENGTH, 'Cabin',  [ (None, 'steerage')
#               ('A','A'),('B','B'),('C','C'),('D','D'),('E','E'),('F','F'),('G','G'),('T','steerage')])
#       
# List: [tuple, tuple]
# tuple = (search string, new value)
######
class cleanRebucketByInclude(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REBUCKET_BY_INCLUDE
        return
  
    def execute(self, df, project, cleaningRules=None, usingRegex=False):
        if self.name not in df:
            return None
        for rule in self.value:
            string, newValue = tuple(rule)
            #mlUtility. traceLog(('Rule, string, newvalue = ',string, newValue)
            if string is None:
                df[self.name].fillna(newValue, inplace = True)
            else:
                if newValue is None or len(newValue)==0:
                    newValue = string
                column = df[self.name]
                mask = column.str.contains(string, regex=usingRegex)
                df.loc[ mask , self.name]  = newValue        
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_FIX_CASE
        return
    
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
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
    def __init__ (self, name, value=None,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_ZERO_FILL
        return
   
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        #print ('zero fill', type(self.value), self.value)
        if self.value is None:
            df[self.name].fillna(0, inplace = True)
        elif type(self.value) is str:
            if self.value == 'mean':
                df[self.name].fillna(df[self.name].mean(), inplace = True)
            elif self.value == 'median':
                df[self.name].fillna(df[self.name].median(), inplace = True)
            elif self.value == 'max':
                df[self.name].fillna(df[self.name].max(), inplace = True)
            elif self.value == 'min':
                df[self.name].fillna(df[self.name].min(), inplace = True)
        else:
            df[self.name].fillna(self.value, inplace = True)
            
            
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REMOVE_ITEMS_BELOW
        return
   
    def execute(self, df, project,cleaningRules=None):
        if self.name not in df:
            return None
        if project.mergedTrainingAndTest:
            mask = (df[self.name] <= self.value) & df[project.IsTrainingSet]
        else:
            mask = (df[self.name] <= self.value)
            
        df.drop(df[mask].index, inplace=True)
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REMOVE_ITEMS_ABOVE
        return
 
    
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
       
        if project.mergedTrainingAndTest:
            mask = (df[self.name] >= self.value) & df[project.IsTrainingSet]
        else:
            mask = (df[self.name] >= self.value)
            
        df.drop(df[mask].index, inplace=True)
        
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_REMOVE_ITEMS_EQUAL
        return
   
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        if self.value:
            if project.mergedTrainingAndTest:
                mask = (df[self.name] == self.value) & df[project.IsTrainingSet]
            else:
                mask = (df[self.name] == self.value)
        else:
            if project.mergedTrainingAndTest:
                mask = (df[self.name].isnull()) & df[project.IsTrainingSet]
            else:
                mask = (df[self.name].isnull())            

        df.drop(df[mask].index, inplace=True)
        
        return None



class cleanKeepItemsEqual(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_KEEP_ITEMS_EQUAL
        return
   
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        if type(self.value) is not list:
            keepList = [self.value]
        else:
            keepList = self.value
            
            
        if self.name in df:
            valList = df[self.name].unique().tolist()
        else:
            mlUtility.errorLog('Unable to Keep list for {} and values {}'.format(self.name, self.value))
            return None

            
        for toDrop in valList:
            if toDrop in keepList:
                pass
            else:
                if project.mergedTrainingAndTest:
                    mask = (df[self.name] == toDrop) & df[project.IsTrainingSet]
                else:
                    mask = (df[self.name] == toDrop)
            
                df.drop(df[mask].index, inplace=True)
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_CONVERT_DATATYPE
        return
   
    def execute(self, df, project, cleaningRules=None):
        df[self.name] = df[self.name].astype(self.value)
        return None



class cleanConvertToBoolean(object):
    def __init__ (self, name, value=None, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_CONVERT_TO_BOOLEAN
        return
   
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        if self.value is None:  
            trueValue = True
        else:
            trueValue = self.value
    
        df[self.name] =  (df[self.name]==trueValue).astype(int)
        return None



   
# Used for keys (A to Z, a to z, ignores the rest)
class cleanConvertColumnToInteger(object):
    def __init__ (self, name, value, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_CONVERT_COLUMN_TO_INTEGER
        return
   
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        df[self.name], _ = pd.factorize(df[self.name])
        #df[self.name] = df[self.name].apply(lambda x: mlUtility.convertAlphaTextToInteger(x))
        df[self.name] = df[self.name].astype('int64')
        return None




# To be used only on columns with fewer unique values, say 20% of col size
class cleanConvertCategoryToInteger(object):
    def __init__ (self, name, value, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_CONVERT_CATEGORY_TO_INTEGER
        return
   
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        df[self.name], _ = pd.factorize(df[self.name])
        df[self.name] = df[self.name].astype('int64')

        return None


class cleanSetCatgory(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_SET_CATEGORY_DATATYPE
        return
   
    def execute(self, df, project, cleaningRules=None):
        if self.name not in df:
            return None
        #mlUtility. traceLog(('\n\nThe Categories',self.name, self.value)
        #df[self.name] = df[self.name].astype('category',categories=self.value)
        if self.value is None:
            self.value = list(set(df[self.name]))
            
        df[self.name] = df[self.name].astype(pd.api.types.CategoricalDtype(categories = self.value))
        
        #df[self.name] = pd.Categorical([self.name], categories=self.value)
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
class cleanDropLowImportanceAndCorr(object):
    def __init__ (self, name, value=None, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_DROP_LOW_FEATURE_IMPORTANCE_AND_CORR
        return
 
    
    def execute(self, df, project, cleaningRules=None):
        if type(self.name) is not list:
            if self.name in df:
                df.drop(self.name, axis=1, inplace=True)
        else:
            for name in self.name:
                if name in df:
                    df.drop(name, axis=1, inplace=True)
                else:
                    mlUtility.errorLog( 'Error: Columns {} not found to drop'.format(name))
        return None


class cleanDropNoCorrColumn(object):
    def __init__ (self, name, value=None, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_DROP_NO_CORR_COLUMN
        return
 
    
    def execute(self, df, project, cleaningRules=None):
        if type(self.name) is not list:
            if self.name in df:
                df.drop(self.name, axis=1, inplace=True)
        else:
            for name in self.name:
                if name in df:
                    df.drop(name, axis=1, inplace=True)
                else:
                    mlUtility.errorLog( 'Error: Columns {} not found to drop'.format(name))
        return None


class cleanDropColumn(object):
    def __init__ (self, name, value=None, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_DROP_COLUMN
        return
 
    
    def execute(self, df, project, cleaningRules=None):
        if type(self.name) is not list:
            if self.name in df:
                df.drop(self.name, axis=1, inplace=True)
        else:
            for name in self.name:
                if name in df:
                    df.drop(name, axis=1, inplace=True)
                else:
                    mlUtility.errorLog( 'Error: Columns {} not found to drop'.format(name))
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
    def __init__ (self, name, value, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_NEW_FEATURE
        return
 
    def execute(self, df, project, cleaningRules=None):
        eq = equationPrep(df, self.name, self.value, indicator=False)  
        exec(eq, {'__builtins__' : None }, {'df' : df })
        return None



class cleanNewFeatureDateDifferenceMonths(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_NEW_FEATURE_DATE_DIFFERENCE_MONTHS
        return
 
    def execute(self, df, project, cleaningRules=None):
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_NEW_INDICATOR_VARIABLE
        return
 
    def execute(self, df, project, cleaningRules=None):
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
    def __init__ (self, name=None, value=None,forPredict=False):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_DROP_DUPLICATES
        
    def execute(self, df, project, cleaningRules=None):
        # drop duplicates
        
        if project.mergedTrainingAndTest:
            mask = (df.duplicated() & df[project.IsTrainingSet])
            df.drop(df[mask].index, inplace=True)
        else:
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
#       project.addManualRuleForDefault(CLEANDATA_GROUPBY_ROLLUP, 
#                                        ['CustomerID','Sales'],  # name[0]=List of names to group by
#                                        { 'total_sales' : 'sum', 'avg_product_value' : 'mean' })
# params
#           name = [0]=list of of column names to group by from the intermediate table
#                  [1]= the column to aggegrate
#           value = dictoinary of new column names and function to agg 
#                 (values are 'nunique', 'count', 'sum', 'mean', 'min', 'max' )
#
#
######
class cleanRollup(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_GROUPBY_ROLLUP
        
    def execute(self, df, project, cleaningRules=None):        
        # Roll up  data
        # get the agg functions (mean, max, etx)
        toAgg = [val for key, val in self.value.items()]
        
        # run the group by
        newCol = df.groupby(self.name[0])[self.name[1]].agg( toAgg )
    
        # rename the columns from the defaults
        toRename = {val: key for key, val in self.value.items()}
        newCol.rename( columns=toRename, inplace=True)
        
        # save the new rollup in the rollup list
        cleaningRules.rollups.append(pd.DataFrame(newCol))
        #mlUtility. traceLog((cleaningRules.rollups)
        return None


######
#
# Description: safe the rollup table to use for predicts.
#
#
# example:
#        update [trainingTable with 'table2'] 
#                for [trainingTable.ColNameToUpdate, 
#                with table2.ColNameToUpdate] 
#                where [trainingTable.colValue equals table2.colKey]
#
#       project.addManualRuleForDefault(ed.CLEANDATA_UPDATE_FROM_TABLE2,
#                                         ['Survival Data','Mean Ages'],
#                                         [['Age','Mean Ages'],['Name','Name']], forPredict=True)
#
######
class cleanUpdateFromTable2(object):
    def __init__ (self, name, value, forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.table2 = None
        self.ruleName = CLEANDATA_UPDATE_FROM_TABLE2
        
    def execute(self, df, project, cleaningRules=None):        
        #        update [training_table with 'table2'] 
        #                for [trainingTable.ColNameToUpdate, 
        #                with table2.ColNameToUpdateWith] 
        #                where [trainingTable.colValue equals table2.colKey]
        #
        # Make the table presist
        if self.table2 is None:
            table2 = cleaningRules.project.preppedTablesDF[self.name]
        forVals = self.value[0]
        whereVals = self.value[1]


#        project.addManualRuleForDefault(ed.CLEANDATA_GROUPBY_ROLLUP, ['Name', 'Age'], { 'Mean Ages' : 'mean' }, forPredict=False)
#        project.addManualRuleForDefault(ed.CLEANDATA_JOIN_ROLLUPS,'Mean Ages', None, forPredict=False)
        
        # update [training_table with 'table2'] for [trainingTable.ColNameToUpdate, 
        #                                    with table2.ColNameToUpdateWith] 
        #                                    where [trainingTable.colValue equals table2.colKey]
#        project.addManualRuleForDefault(ed.CLEANDATA_UPDATE_FROM_TABLE2,'Mean Ages',
#                                     [['Age','Mean Ages'],['Name','GroupName']], forPredict=True)

        
        
        trainingTable_ColNameToUpdate = forVals[0] # Age
        table2_ColNameToUpdateWith = forVals[1] # Mean Ages
        trainingTable_colValue = whereVals[0] # Name
        table2_colKey = whereVals[1] # GroupName
        
        # for each value in the training table
        for rowKey in table2.index:
            updateValue = table2[table2_colKey][rowKey]
            
            mlUtility.runLog ('   -Updating {} with {} where {}={} and {} is Null'.format(trainingTable_ColNameToUpdate,updateValue, 
                                                trainingTable_colValue,rowKey,trainingTable_ColNameToUpdate))
            
            #
            #
            
            mask = (df[trainingTable_colValue] == rowKey) & (np.isnan(df[trainingTable_ColNameToUpdate]))
            #mask = (df[trainingTable_colValue] == rowKey) & (df[trainingTable_ColNameToUpdate] == np.nan)
            #mlUtility. traceLog(('mask=',mask)                                              
            df.loc[mask, trainingTable_ColNameToUpdate ] = updateValue
            #df.to_csv('GroupTest.csv')
        return None
        #df.loc[ (df[self.name] > lower) & (df[self.name] <= upper), self.name]  = newValue
                #mask = column.str.contains(string, regex=usingRegex)
                #df.loc[ mask , self.name]  = newValue        


######
#
# Description: Create an intermediate group as rollup to be further rolluped
#
# example:
#    project.addManualRuleForDefault(CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP, [['CustomerID','InvoiceNo'],'Sales'],
#        { 'cart_value' : 'sum' } )
#                                                                                                  'max_cart_value' : 'max'})
#    args:
#           name = [0]=list of of column names to group by from the intermediate table
#                  [1]= the column to aggegrate
#           value = dictoinary of new column names and function to agg (values are 'nunique', 'count', 'sum', 'mean', 'min', 'max' )
#
######
class cleanIntermediaryGroup(object):
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP
        
    def execute(self, df, project, cleaningRules=None):


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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP
        
    def execute(self, df, project, cleaningRules=None):
              
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_JOIN_ROLLUPS
        
    def execute(self, df, project, cleaningRules=None):
        if len(cleaningRules.rollups) > 1:
            main = cleaningRules.rollups.pop()
            cleaningRules.project.preppedTablesDF[self.name] = main.join(cleaningRules.rollups)
        elif len(cleaningRules.rollups)==1:
            cleaningRules.project.preppedTablesDF[self.name] = cleaningRules.rollups.pop()
        else:
            mlUtility.raiseError('No Rollups to Join')
            
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_PCA_DIMENSIONALITY_REDUCTION
        
    def execute(self, df, project, cleaningRules=None):
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION
        
    def execute(self, df, project, cleaningRules=None):
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_JOIN_TABLES
        
    def execute(self, df, project, cleaningRules=None):

        # test for file names
        for x in self.name:
            if x not in cleaningRules.project.preppedTablesDF:
                mlUtility.raiseError('Table "{}" not found'.format(x))
                return None
        if len(self.name) < 2:
            mlUtility.raiseError('Not enough tables to join. sent='.format(selfname))
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
    def __init__ (self, name, value,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_SET_BATCH_TABLES
        
    def execute(self, df, project, cleaningRules=None):
        for n in self.name:
            if n not in cleaningRules.project.preppedTablesDF:
                mlUtility.raiseError('Table not found: '+n)
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
    def __init__ (self, name=None, value=None,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_DROP_NA
        return
   
    def execute(self, df, project, cleaningRules=None):
        if project.mergedTrainingAndTest:
            mask = (df.isna() & df[project.IsTrainingSet])
            df.drop(df[mask].index, inplace=True)
        else:
            df.dropna(inplace=True)
        return None
  

# This does nothing (used when just a recommendation)
class cleanNoOpp(object):
    def __init__ (self, name=None, value=None,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_NO_OPP
        return
   
    def execute(self, df, project, cleaningRules=None):
       return None
  


class cleanDropNAForColumn(object):
    def __init__ (self, name, value=None,forPredict=True):
        self.name = name
        self.value = value
        self.forPredict = forPredict
        self.ruleName = CLEANDATA_DROP_NA_FOR_COLUMN
        return
   
    def execute(self, df, project, cleaningRules=None):

        if project.mergedTrainingAndTest:
            mask = (df[self.name].isna() & df[project.IsTrainingSet])
            df.drop(df[mask].index, inplace=True)
        else:
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
    


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=DataConversionWarning)
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
        

def doClean(df, project, ruleList, isPredict):
    #print ('\n\ndo clean rule list', ruleList,'\n\n')
    for rule in ruleList:
        if rule is not None:
            #print ('\n\nthis is the rule', rule)
            if not rule.forPredict and isPredict:
                theAction = 'Skipping'
            else:
                theAction = 'Running'
            if type(rule.name) is not list:
                if rule.name in df.columns:
                    mlUtility.runLog( '{} Rule {} for Column {} with value {}'.format(theAction, rule.ruleName, rule.name, rule.value))
                elif rule.name is not None:
                    if rule.value is None:
                        mlUtility.runLog( '{} Rule {} '.format(theAction, rule.ruleName))
                    else:
                        mlUtility.runLog( '{} Rule {} for {} with value {}'.format(theAction, rule.ruleName, rule.name, rule.value))
                else:
                    mlUtility.runLog( '{} Rule {} '.format(theAction, rule.ruleName))
            else:
                if rule.value is None:
                    mlUtility.runLog( '{} Rule {} for {}'.format(theAction, rule.ruleName,rule.name))
                else:
                    mlUtility.runLog( '{} Rule {} for {} with value {}'.format(theAction, rule.ruleName, rule.name, rule.value)) 
            #sizeBefore = df.shape    
            # Only execute if for training (mostly true)
            if not rule.forPredict and isPredict:
                pass
            else:
                rule.execute(df, project, cleaningRules)   
            #mlUtility. traceLog(('Size Before Rule {} and After {}'.format(sizeBefore, df.shape ))



def cleanData(df, cleaningRules, isPredict=False):

    # Execute rules
    if cleaningRules is not None:
        doClean(df, cleaningRules.project, cleaningRules.rules, isPredict)
    return 

def getRule(functionName, columnName, value , df=None, forPredict=True):
    if functionName == CLEANDATA_MARK_MISSING:
        cleanFunction = cleanMarkMissing(columnName, value, forPredict=forPredict)        
    elif functionName == CLEANDATA_FIX_CASE:
        cleanFunction =  cleanFixCase(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_DROP_DUPLICATES:
        cleanFunction =  cleanDropDuplicates(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_FIX_SKEW:
        cleanFunction =  cleanFixSkew(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_ZERO_FILL:
        cleanFunction =  cleanZeroFill(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REBUCKET:
        cleanFunction = cleanRebucket(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REBUCKET_WHERE:
        cleanFunction = cleanRebucketWhere(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REBUCKET_TO_BINARY:
        cleanFunction = cleanRebucketBinary(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REBUCKET_BY_RANGE:
        cleanFunction = cleanRebucketByRange(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REBUCKET_BY_INCLUDE:
        cleanFunction = cleanRebucketByInclude(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_CONVERT_CATEGORY_TO_INTEGER:
        cleanFunction = cleanConvertCategoryToInteger(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_CONVERT_COLUMN_TO_INTEGER:
        cleanFunction = cleanConvertColumnToInteger(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_DROP_ID_FIELD:
        cleanFunction =  cleanDropColumn(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REMOVE_ITEMS_BELOW:
        cleanFunction = cleanRemoveItemsBelow(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REMOVE_ITEMS_ABOVE:
        cleanFunction = cleanRemoveItemsAbove(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_REMOVE_ITEMS_EQUAL:
        cleanFunction = cleanRemoveItemsEqual(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_KEEP_ITEMS_EQUAL:
        cleanFunction = cleanKeepItemsEqual(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_NEW_FEATURE:
        cleanFunction = cleanNewFeature(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_NEW_FEATURE_DATE_DIFFERENCE_MONTHS:
        cleanFunction = cleanNewFeatureDateDifferenceMonths(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_DROP_COLUMN:
        cleanFunction = cleanDropColumn(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_DROP_NO_CORR_COLUMN:
        cleanFunction = cleanDropNoCorrColumn(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_DROP_LOW_FEATURE_IMPORTANCE_AND_CORR:
        cleanFunction = cleanDropLowImportanceAndCorr(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_NEW_INDICATOR_VARIABLE:
        cleanFunction = cleanNewIndicatorVariable(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_NUMERIC_MARK_MISSING:
        cleanFunction = cleanNumericMarkMissing(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_CONVERT_DATATYPE:
        cleanFunction = cleanConvertDatatype(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_CONVERT_TO_BOOLEAN:
        cleanFunction = cleanConvertToBoolean(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_STRIP_TO_ALPHA:
        cleanFunction = cleanStripToAlpha(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_DROP_NA:
        cleanFunction = cleanDropNA(forPredict=forPredict)
    elif functionName == CLEANDATA_DROP_NA_FOR_COLUMN:
        cleanFunction = cleanDropNAForColumn(columnName, forPredict=forPredict)
    elif functionName == CLEANDATA_SET_CATEGORY_DATATYPE:
        if value is not None:
            columns = list(set(value))
        else:
            columns = None
        #mlUtility. traceLog(('\n\nClean set category', columnName, columns)
        cleanFunction = cleanSetCatgory(columnName, columns, forPredict=forPredict)

    # Data shaping functions
    elif functionName == CLEANDATA_GROUPBY_ROLLUP:
        cleanFunction = cleanRollup(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_UPDATE_FROM_TABLE2:
        cleanFunction = cleanUpdateFromTable2(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_INTERMEDIARY_GROUPBY_ROLLUP:
        cleanFunction = cleanIntermediaryGroup(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_ROLLUP_INTERMEDIARY_GROUPBY_ROLLUP:
        cleanFunction = cleanRollupIntermediary(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_JOIN_ROLLUPS:
        cleanFunction = cleanJoinRollup(columnName, value, forPredict=forPredict)


    elif functionName == CLEANDATA_PCA_DIMENSIONALITY_REDUCTION:
        cleanFunction = cleanPCADimensionalityReduction(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_THRESHOLD_DIMENSIONALITY_REDUCTION:
        cleanFunction = cleanThresholdDimensionalityReduction(columnName, value, forPredict=forPredict)
        
    elif functionName == CLEANDATA_JOIN_TABLES:
        cleanFunction = cleanJoinTables(columnName, value, forPredict=forPredict)
    elif functionName == CLEANDATA_SET_BATCH_TABLES:
        cleanFunction = cleanSetBatchTables(columnName, value, forPredict=forPredict)
    
    else:
        cleanFunction = cleanNoOpp(columnName, value, forPredict=forPredict)
        return cleanFunction
    #print ('get\n\nRule', functionName, columnName, cleanFunction)
    return cleanFunction
        


class cleaningRules(object):

    def __init__ (self, project, explore):
        self.rules = []
        self.project = project
        self.rollups = []
        self.intermediary = None
        self.df = explore.dataFrame
        if project.runAutoFeaturesMode:
            if project.dropDuplicates:
                self.addManualRule(CLEANDATA_DROP_DUPLICATES, None, None, forPredict=False )
            for columnName, paramater in explore.recommendations.items():          
                for functionName, value in paramater.items():
                    # Pull out the name in the function name to get the function
                    i = functionName.find(CLEANDATA_BREAK)
                    if i == -1:
                        functionNameSub = functionName
                    else:
                        functionNameSub = functionName[:i]
                    if functionNameSub in exploreAutoFunctions:
                        #print ('>>>>>add manual rules',functionNameSub, columnName, value)
                        self.addManualRule(functionNameSub, columnName, value, forPredict=True )
                        
        return


    def addManualRule(self, functionName, columnName, value , forPredict=True):
        cleanFunction = getRule(functionName, columnName, value , self.df, forPredict)
        self.rules.append(cleanFunction)
        return cleanFunction
             
       
       
            
    def addAndRunManualRule(self, df, functionName, columnName, value, forPredict=True ):
        rule = self.addManualRule(functionName, columnName, value, forPredict=forPredict)
        doClean(df, self.project, [rule], forPredict)
        
      



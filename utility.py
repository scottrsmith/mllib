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
import datetime

runLog = True


def removePlus(value):
    return value.split('+')[0]
    

def openLogs():
    pass

def closeLogs():
    pass


def runLog (message):
    if runLog:
        print ('{} @ {}'.format(message, datetime.datetime.now())) 

def errorLog (message):
    print ('ERROR: {} @ {}'.format(message, datetime.datetime.now()))
    
    
def activityLog (message):
    print ('ACTVITY: {} @ {}'.format(message, datetime.datetime.now()))
    

def raiseError (message):
    errorLog(message)

    
def table(rows, margin=0, columns=[]):
    """
    Return string representing table content, returns table as string and as a list of strings.
    It is okay for rows to have different sets of keys, table will show union of columns with
    missing values being empty spaces.
    :param rows: list of dictionaries as rows
    :param margin: left space padding to apply to each row, default is 0
    :param columns: extract listed columns in provided order, other columns will be ignored
    :return: table content as string and as list
    """
    def projection(cols, columns):
        return [(x, cols[x]) for x in columns if x in cols] if columns else cols.items()
    def row_to_string(row, columns):
        values = [(row[name] if name in row else "").rjust(size) for name, size in columns]
        return "|%s|" % ("|".join(values))
    def header(columns):
        return "|%s|" % ("|".join([name.rjust(size) for name, size in columns]))
    def divisor(columns):
        return "+%s+" % ("+".join(["-" * size for name, size in columns]))
    data = [dict([(str(a), str(b)) for a, b in row.items()]) for row in rows]
    cols = dict([(x, len(x) + 1) for row in data for x in row.keys()]) if data else {}
    for row in data:
        for key in row.keys():
            cols[key] = max(cols[key], len(row[key]) + 1)
    proj = projection(cols, columns) # extract certain columns to display (or all if not provided)
    table = [divisor(proj), header(proj), divisor(proj)] + \
        [row_to_string(row, proj) for row in data] + [divisor(proj)]
    table = ["%s%s" % (" " * margin, tpl) for tpl in table] if margin > 0 else table
    table_text = "\n".join(table)
    return (table_text, table)

def printAsTable(rows, margin=0, columns=[]):
    """
    Print table in console for list of rows.
    """
    txt, _ = table(rows, margin, columns)
    print (txt)
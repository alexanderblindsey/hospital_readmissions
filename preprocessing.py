#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:42:50 2020

@author: alexanderlindsey

Preprocessing diabetic_data.csv into the dataset used for initial
data visualizations. 
"""

import pandas as pd
import numpy as np
from numpy import random
from tqdm import tqdm

# read data
PATH = 'Data/Raw/diabetic_data.csv'
df = pd.read_csv(PATH)

# NaN represented as ? in raw file
df.replace('?', np.nan, inplace=True) 
df.drop(['encounter_id'], axis=1, inplace=True) # encounter_id is redundant - 
                                                # each encounter_id and index 
                                                # are unique
# make readmitted binary
df['readmitted'].replace({'<30':'YES', '>30':'YES'}, inplace=True)

# drop duplicates
df.drop_duplicates(subset= ['patient_nbr'], keep = 'first', inplace=True)

# patient_nbr becomes redundant
df.drop(['patient_nbr'], axis=1, inplace=True) 

# impute missing values
df.rename(columns={'diag_1':'diagnosis'}, inplace=True) 
max_race = df['race'].value_counts().idxmax() # missing values replaced with 
                                              # most observed value of that feature
max_diag = df['diagnosis'].value_counts().idxmax()

for i in tqdm(df.index):
    current_race = df.at[i, 'race']
    current_diag = df.at[i, 'diagnosis']
    
    if pd.isnull(current_race) == True:
        df.at[i, 'race'] = max_race

    if pd.isnull(current_diag) == True:
        df.at[i, 'diagnosis'] = max_diag
        
# drop initial columns
cols_to_drop = ['weight',
                'payer_code',
                'medical_specialty',
                'diag_2',
                'diag_3',
                'examide',
                'glimepiride-pioglitazone',
                'citoglipton']

df.drop(cols_to_drop, axis=1, inplace=True)

# categorizing diagnoses into 18 larger categories
for i in df.index:
    d = df.at[i, 'diagnosis']
    
    try:
        d = float(df.at[i, 'diagnosis'])
        
    except ValueError:
        df.at[i, 'diagnosis'] = 'other'
    
    else:
        if d>=1 and d<=139:
            df.at[i, 'diagnosis'] = 'infections/parasitic'
        elif d>=140 and d<=239:
            df.at[i, 'diagnosis'] = 'neoplasms'
        elif d>=240 and d<=279:
            df.at[i, 'diagnosis'] = 'endocrine/nutrition/immunity'
        elif d>= 280 and d<=289:
            df.at[i, 'diagnosis'] = 'blood'
        elif d>=290 and d<=319:
            df.at[i, 'diagnosis'] = 'mental disorders'
        elif d>=320 and d<=329:
            df.at[i, 'diagnosis'] = 'nervous system'
        elif d>=390 and d<=459:
            df.at[i, 'diagnosis'] = 'circulatory'
        elif d>=460 and d<=519:
            df.at[i, 'diagnosis'] = 'respiratory'
        elif d>=520 and d<=579:
            df.at[i, 'diagnosis'] = 'digestive'
        elif d>=580 and d<=629:
            df.at[i, 'diagnosis'] = 'genitourinary'
        elif d>=630 and d<=679:
            df.at[i, 'diagnosis'] = 'pregnancy/childbirth'
        elif d>=680 and d<=709:
            df.at[i, 'diagnosis'] = 'skin/subcutaneous tissue'
        elif d>=710 and d<=739:
            df.at[i, 'diagnosis'] = 'musculoskeletal'
        elif d>=740 and d<=759:
            df.at[i, 'diagnosis'] = 'congenital anomalies'
        elif d>=760 and d<=779:
            df.at[i, 'diagnosis'] = 'perinatal'
        elif d>=780 and d<=799:
            df.at[i, 'diagnosis'] = 'ill-defined'
        else:
            df.at[i, 'diagnosis'] = 'injury/poisoning'
            
            
# gender has two 'unknown' values - randomly assign to male or female
x = random.randint(2)

for i in df.index:
    current_gender = df.at[i, 'gender']
    
    if current_gender == 'Unknown/Invalid':
        if x == 0:
            df.at[i, 'gender'] = 'Male'
        else:
            df.at[i, 'gender'] = 'Female'         


# reducing A1C other test results
for i in df.index:
    current_a1c = df.at[i, 'A1Cresult']
    current_glu = df.at[i, 'max_glu_serum']
    
    if current_a1c in ['>8', '>7']:
        df.at[i, 'A1Cresult'] = 'high'
    
    if current_glu in ['>200', '>300']:
        df.at[i, 'max_glu_serum'] = 'high'



        

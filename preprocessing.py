#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:42:50 2020

@author: alexanderlindsey

Preprocessing diabetic_data.csv into the dataset used for modelling 
and visualizations.
"""

import pandas as pd
import numpy as np
from numpy import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

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


# reducing possible values for A1C and other test results
for i in df.index:
    current_a1c = df.at[i, 'A1Cresult']
    current_glu = df.at[i, 'max_glu_serum']
    
    if current_a1c in ['>8', '>7']:
        df.at[i, 'A1Cresult'] = 'high'
    
    if current_glu in ['>200', '>300']:
        df.at[i, 'max_glu_serum'] = 'high'

# save - for initial viz
df.to_csv(r'Data/Processed/df1.csv')

# find changes in patient drug regimen
df['drug_changes'] = 0
columns = ['gender',
           'diabetesMed',
           'admission_type_id',
           'admission_source_id',
           'time_in_hospital',
           'num_procedures',
           'number_inpatient',
           'max_glu_serum',
           'A1Cresult',
           'change',
           'age',
           'discharge_disposition_id',
           'number_outpatient',
           'race',           
           'num_medications',
           'number_emergency',
           'number_diagnoses',
           'num_lab_procedures',
           'diagnosis']
drug_cols = [col for col in df.columns.drop(['readmitted']) if col not in columns]

for i in tqdm(df.index):
    for c in drug_cols:
        if df.at[i, c] == 'Up':
            df.at[i, 'drug_changes'] = df.at[i, 'drug_changes'] + 1
        elif df.at[i, c] == 'Down':
            df.at[i, 'drug_changes'] = df.at[i, 'drug_changes'] + 1

df.drop(drug_cols, axis=1, inplace=True) # drop drug cols - responsible for too
                                         # many features in one-hot-encoded df

# reduce possible values for admission_source_id, admission_type_id, and
# discharge_disposition_id

for i in tqdm(df.index):
    current_admin_source = df.at[i, 'admission_source_id']
    current_admin_type = df.at[i, 'admission_type_id']
    current_dis = df.at[i, 'discharge_disposition_id']
    
    # admission_source_id
    if current_admin_source in [2, 3]: # lump all referalls
        df.at[i, 'admission_source_id'] = 1 
    
    elif current_admin_source in [5, 6, 10, 18, 19, 22, 25, 26]: # lump all transfers and the one readmission from 
        df.at[i, 'admission_source_id'] = 4                      # the same health agency
        
    elif current_admin_source in [12, 14, 23, 24]: # lump all baby deliveries
        df.at[i, 'admission_source_id'] = 11
        
    elif current_admin_source in [15, 17, 20, 21]: # lump all not available, not mapped, unknown, and NULL
        df.at[i, 'admission_source_id'] = 9 
    
    
    # admission_type_id
    if current_admin_type in [6, 8]: # lump all not available, unknown, NULL
        df.at[i, 'admission_type_id'] = 5
    
    elif current_admin_type in [2, 7]: # lump emergency, urgent care, and trauma center
        df.at[i, 'admission_type_id'] = 1 
    
    
    # discharge_disposition_id
    if current_dis == 5: # lump discharged to home and discharged to home with home health agency
        df.at[i, 'discharge_disposition_id'] = 1 
        
    elif current_dis in [3, 4, 5, 6, 8, 10, 15, 17, 22, 23, 24, 16, 30, 27, 28, 29]: # lump discharged to another facility
        df.at[i, 'discharge_disposition_id'] = 2
        
    elif current_dis in [19, 20, 21]: # lump expired patients
        df.at[i, 'discharge_disposition_id'] = 11 
        
    elif current_dis == 13: # lump both hospice
        df.at[i, 'discharge_disposition_id'] = 14 
        
    elif current_dis in [25, 26]: # lump Null, unknown, etc
        df.at[i, 'discharge_disposition_id'] = 18

# save - for id columns viz
df.to_csv(r'Data/Processed/df2.csv')

# one hot encode
continuous_cols = []
categorical_cols = []
for c in df.columns:
    if df[c].dtype=='int64':
        if 'id' not in c:
            continuous_cols.append(c)
        else:
            categorical_cols.append(c)
    else:
        categorical_cols.append(c)
df_ohe = pd.get_dummies(data=df, columns=categorical_cols, drop_first=True)

# splitting
X_train, X_test, y_train, y_test = train_test_split(df_ohe.drop(['readmitted_YES'], axis=1), 
                                                    df_ohe['readmitted_YES'],
                                                    train_size=.8,
                                                    shuffle=True,
                                                    random_state=1)

# scaling
ct = ColumnTransformer(transformers=[('scaler', StandardScaler(), continuous_cols)],
                       remainder='passthrough')

X_train = pd.DataFrame(ct.fit_transform(X_train),
                       columns=X_train.columns)
X_test = pd.DataFrame(ct.transform(X_test),
                      columns=X_test.columns)

df_ohe.to_csv(r'Data/Processed/df_ohe.csv')
X_train.to_csv(r'Data/Processed/X_train.csv')
X_test.to_csv(r'Data/Processed/X_test.csv')
y_train.to_csv(r'Data/Processed/y_train.csv')
y_test.to_csv(r'Data/Processed/y_test.csv')
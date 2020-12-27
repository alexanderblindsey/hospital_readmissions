#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:29:08 2020

@author: alexanderlindsey

Preprocessing IDs_mapping.csv
"""

import pandas as pd

# read data
PATH = 'Data/Raw/IDs_mapping.csv'

ids = pd.read_csv(PATH)

ids = ids[(ids['description']!='description')] 

# merge
ids = pd.merge(ids.copy()[40:].rename(columns={'description':'admission_source_id',
                                               'admission_type_id':'values'}),
              (pd.merge(ids.copy()[0:8].rename(columns={'description':'admission_type_id',
                                                        'admission_type_id':'values'}),
                        ids.copy()[9:39].rename(columns={'description':'discharge_disposition_id',
                                                         'admission_type_id':'values'}),
                           how='outer',
                           left_on='values',
                           right_on='values')),
              how='outer',
              left_on='values',
              right_on='values').reset_index(drop=True)

# some NAN should actually be NULL
ids.loc[15, 'admission_source_id'] = 'NULL'
ids.loc[5, 'admission_type_id'] = 'NULL'
ids.loc[16, 'discharge_disposition_id'] = 'NULL'

# save
ids.to_csv(r'Data/Processed/IDs_mapping_processed.csv', index_label=False)
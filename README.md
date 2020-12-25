# Predicting Hospital Readmission

December 21, 2020

## Background

In this investigation, I use the data collected in [Strack et. al. (2014)](https://www.hindawi.com/journals/bmri/2014/781670/) to predict the readmission of patients admitted to a hospital or integrated delivery network in the United States from 1999-2008. The data contains 101,766 hospital visits for 71,518 unique patients. The dataset contains patient information, such as their race, gender, and age, alongside variables relating to the nature of their admission to the hospital, such as their diagnoses, time in hospital, test results, etc.

## Problem

A readmission occurs when a patient has been readmitted to the hospital within 30 days of an earlier hospital stay. The economic cost of hospital readmissions is considerably large - readmission of elderly patients costs the federal health insurer, Medicare, approximately [$15 billion dollars per year](https://www.healthstream.com/resources/blog/blog/2020/06/02/the-economic-emotional-cost-of-hospital-readmissions#:~:text=The%20cost%20of%20hospital%20readmissions,the%20program%20received%20readmissions%20penalties.).

The Center for Medicare & Medicaid Services (CMS) penalizes hospitals for patient readmission, reducing hospital revenue. Additionally, patients and their families suffer an emotional, physical, and financial toll when they are readmitted. 

Models similair to the one(s) created in this investigation could be used by hospitals and medical insurers to predict hospital readmission, driving down costs. 

## Goal

The goal of this investigation is to create a classification model using supervised machine learning that predicts if a patient will be readmitted or not. Models are optimized for  accuracy.


## Data Description

One problem with this dataset is that there are _a lot_ of categorical features - one-hot encoding the dataset with minimal feature engineering results in over 2,500 columns. While this may also impact model accuracy, training models with so many columns and tens of thousands of observations is very computationally intensive and therefore impractical. 

Below is a list of features and their descriptions. 
- `encounter_id`: Unique identifier of an encounter.
- `patient_nbr`: Unique identifier of a patient.
- `race`: Caucasian, Asian, African American, Hispanic, and other.
- `gender`: male, female, and unknown/invalid.
- `age`: Grouped in 10-year intervals: 0, 10), 10, 20), …, 90, 100).
- `weight`: Weight in pounds.
- `admission_type_id`: Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available.
- `discharge_disposition_id`: Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available.
- `admission source_id`: Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital.
- `time_in_hospital`: Integer number of days between admission and discharge.
- `payer_code`: Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay.
- `medical_specialty`: Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon.
- `num_lab_procedures`: Number of lab tests performed during the encounter.
- `numb_procedures`: Number of procedures (other than lab tests) performed during the encounter.
- `num_medications`: Number of distinct generic names administered during the encounter.
- `number_outpatient`: Number of outpatient visits of the patient in the year preceding the encounter.
- `number_emergency`: Number of emergency visits of the patient in the year preceding the encounter.
- `number_inpatient`: Number of inpatient visits of the patient in the year preceding the encounter.
- `diag_1`: The primary diagnosis (coded as first three digits of ICD9); 848 distinct values.
- `diag_2`: Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values.
- `diag_3`: Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values.
- `number_diagnoses`: Number of diagnoses entered to the system
- `max_glu_serum`: Indicates the range of the result or if the test was not taken. Values: “>200,” “>300,” “normal,” and “none” if not measured.
- `A1Cresult`: Indicates the range of the result or if the test was not taken. Values: “>8” if the result was greater than 8%, “>7” if the result was greater than 7% but less than 8%, “normal” if the result was less than 7%, and “none” if not measured.
- `change`: Indicates if there was a change in diabetic medications (either dosage or generic name). Values: “change” and “no change”.
- `diabetesMed`: Indicates if there was any diabetic medication prescribed. Values: "yes" and "no".
- 24 features for medications: Medications under their generic names: `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `sitagliptin`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, and `metformin- pioglitazone`. The feature indicates whether the drug was prescribed or there was a change in the dosage. _Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed._
- `readmitted`: Days to inpatient readmission. Values: “<30” if the patient was readmitted in less than 30 days, “>30” if the patient was readmitted in more than 30 days, and “No” for no record of readmission
import numpy as np
from sklearn import  model_selection, neighbors
import pandas as pd

# Primary site	1= Extrimity, 2 = Axial
# location	1= Femur, 2= Tibia, 3 = Humerus
# Metastesis 	1 = yes, 0=no, -1 = NA, 2 =others (default)
# Mets at diagNosis	1 = yes, 0=no, -1 = NA, 2 =others (default)
# Mets during treatment	1 = yes, 0=no, -1 = NA, 2 =others (default)
# Mets after treatment	1 = yes, 0=no, -1 = NA, 2 =others (default)
# Tumor size > or < 10cm	1 = <=10cm, 2 = >10cm, -1 = NA
# Enneking stage	1 = IA, 2 = IB, 3= IIA, 4 = IIB, 5 = III
# Serum ALP	1 = Normal, 2 = High
# LMR	1 = <= 2.0, 2 = > 2.0
# Primary surgery 	1 = salvage, 2 = amputation, 3 = conservative, 0 = others, -1 = NA
# Induction Chemotherapy	1 = yes, 0=no, -1 = NA, 2 =others (default)
# Regime DC/DCM/others	1 = DCM/2= DCM/ 0 = others
# Adjuvant Chemo	1 = yes, 0=no, -1 = NA, 2 =others (default)
# Completed Treatment	1 = yes, 0=no, -1 = NA, 2 =others (default)
# -----------------------------------------------------------------------------------
# Length Survival	1 = 0-2, 2 = 2-5, 3 = > 5, -1 = NA

# Test case 1 (random fit patient)
# -----------------------------------
# Primary site	- 2 = Axial
# location - 2= Tibia
# Metastesis - 1 = yes
# Mets at diagNosis	- 0 = no
# Mets during treatment	0 = no
# Mets after treatment	1 = yes
# Tumor size > or < 10cm - 1 = <=10cm
# Enneking stage	1 = IA
# Serum ALP	1 = Normal
# LMR - 1 = <= 2.0
# Primary surgery - 2 = amputation
# Induction Chemotherapy - 1 = yes
# Regime DC/DCM/others - 2= DCM
# Adjuvant Chemo - 1 = yes
# Completed Treatment - 2 =others (default)
# -------------------------------------------------
# [2,2,1,0,0,1,1,1,1,1,2,1,2,1,2]
# Length Survival => 0-2 years


# Test case 2 (low risk patient)
# -----------------------------------
# Primary site	1= Extremity
# location	1= Femur
# Metastesis 	0=no
# Mets at diagNosis	0=no
# Mets during treatment	0=no
# Mets after treatment	0=no
# Tumor size > or < 10cm	1 = <=10cm
# Enneking stage	1 = IA
# Serum ALP	 2 = High
# LMR	1 = <= 2.0
# Primary surgery 	1 = salvage
# Induction Chemotherapy	0=no
# Regime DC/DCM/others	2= DM
# Adjuvant Chemo	0=no
# Completed Treatment	1 = yes
# -------------------------------------------------
# [1,1,0,0,0,0,1,1,2,1,1,1,2,0,1]
# Length Survival => >5 years


# Test case 3 (high risk patient)
# -----------------------------------
# Primary site	2 = Axial
# location	3 = Humerus
# Metastesis 	1 = yes
# Mets at diagNosis	1 = yes
# Mets during treatment	1 = yes
# Mets after treatment	1 = yes
# Tumor size > or < 10cm	2 = >10cm
# Enneking stage	5 = III
# Serum ALP	2 = High
# LMR	2 = > 2.0
# Primary surgery 	2 = amputation
# Induction Chemotherapy	1 = yes
# Regime DC/DCM/others	1 = DCM
# Adjuvant Chemo	1 = yes
# Completed Treatment	0=no
# -------------------------------------------------
# [2,3,1,1,1,1,2,5,2,2,2,1,1,1,0]
# Length Survival => 0-2 years


# Test case 4 (medium risk patient)
# -----------------------------------
# Primary site	2 = Axial
# location	2= Tibia
# Metastesis 	0=no
# Mets at diagNosis	1 = yes
# Mets during treatment	0=no
# Mets after treatment	1 = yes
# Tumor size > or < 10cm	1 = <=10cm
# Enneking stage	3= IIA
# Serum ALP	1 = Normal
# LMR	2 = > 2.0
# Primary surgery 	3 = conservative
# Induction Chemotherapy	0=no
# Regime DC/DCM/others	0 = others
# Adjuvant Chemo	1 = yes
# Completed Treatment	1 = yes
# -------------------------------------------------
# [2,2,0,1,0,1,1,3,1,2,3,0,0,1,1]
# Length Survival => >5 years



#loading the file
filename = 'new_cancer_python.txt'

# reading the file as a CSV
df = pd.read_csv(filename)

#this is used to hadle outliers - (missing data is denoted by '-1')
df.replace('-1', -99999, inplace=True)

#this is used to drop uneeded columns (columns that do not affect the label/ classification)
df.drop(['Age','Gender'], 1, inplace=True)

# setting the x-axis (this column drops the column labeled 'Length_Survival' because that is the classification column)
x = np.array(df.drop(['Length_Survival'],1))

# setting the y-axis (setting 'Length_Survival' as the classification column)
y = np.array(df['Length_Survival'])

# used to train the data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)


# array determines the features of the patient
patient = np.array([1,1,0,0,0,0,1,1,2,1,1,1,2,0,1])
patient = patient.reshape(1,-1)
#
prediction = clf.predict(patient)
print(prediction)


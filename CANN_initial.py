#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:43:33 2024

@author: belguutei
"""


import pyreadr
import numpy as np
import os
import random
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import statsmodels.api as sm
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax, Nadam
from itertools import product


result = pyreadr.read_r('/Users/belguutei/desktop/French.RData')
data = result['freMTPL2freq']

#Random Seed

seed = 100

# Set PYTHONHASHSEED environment variable
os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed for Python random number generator
random.seed(seed)

# Set seed for NumPy random number generator
np.random.seed(seed)

# Set seed for TensorFlow random number generator
tf.random.set_seed(seed)

# Poisson deviance function with safeguards
def poisson_deviance(pred, obs):
    # To prevent division by zero, set a very small value to replace zeros in pred
    epsilon = 1e-10
    pred_safe = np.maximum(pred, epsilon)
    deviance = 200 * (np.sum(pred_safe) - np.sum(obs) + np.sum(obs * np.log(obs / pred_safe)))
    return deviance / len(pred)

# Define a function to plot loss
def plot_loss(history):
    if len(history.history['loss']) > 1:
        df_val = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        })
        df_val = df_val.melt(id_vars=['epoch'], value_vars=['train_loss', 'val_loss'], var_name='variable', value_name='loss')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_val, x='epoch', y='loss', hue='variable')
        plt.title('Loss over Epochs')
        plt.show()
        
        
        
set1 = [0, 6]                   #Optimisers 
set2 = [10, 30]                 #Number of Neurons for the 1st layer
set3 = [5, 25]                  #Number of Neurons for the 2nd layer
set4 = [5, 15]                  #Number of Neurons for the 3rd layer
set5 = [100, 900]               #Epoch
set6 = [5000, 15000]            #Batch Size
set7 = [2, 4]                   #Number of Layers

all_sets = [set1, set2, set3, set4, set5, set6, set7]

# Generate the Cartesian product of all sets
combinations = list(product(*all_sets))

# Create a DataFrame to store combinations
columns = ['Optimizer', 'N1', 'N2', 'N3', 'Epoch', 'Batch', 'Layer']
experiments = pd.DataFrame(combinations, columns=columns)

experiments.loc[128] = [3, 20, 15, 10, 500, 10000, 3]
experiments.loc[129] = [3, 20, 15, 10, 500, 10000, 3]
experiments.loc[130] = [3, 20, 15, 10, 500, 10000, 3]
experiments.loc[131] = [3, 20, 15, 10, 500, 10000, 3]


experiments['Optimizer_coded'] = None
experiments['N1_coded'] = None
experiments['N2_coded'] = None
experiments['N3_coded'] = None
experiments['Epoch_coded'] = None
experiments['Batch_coded'] = None
experiments['Layer_coded'] = None
experiments['loss'] = None
print(experiments)        

m1=(max(set1)+min(set1))/2
s1=(max(set1)-min(set1))/2
m2=(max(set2)+min(set2))/2
s2=(max(set2)-min(set2))/2
m3=(max(set3)+min(set3))/2
s3=(max(set3)-min(set3))/2
m4=(max(set4)+min(set4))/2
s4=(max(set4)-min(set4))/2
m5=(max(set5)+min(set5))/2
s5=(max(set5)-min(set5))/2
m6=(max(set6)+min(set6))/2
s6=(max(set6)-min(set6))/2
m7=(max(set7)+min(set7))/2
s7=(max(set7)-min(set7))/2
experiments['Optimizer_coded']=(experiments['Optimizer']-m1)/s1
experiments['N1_coded'] = (experiments['N1']-m2)/s2
experiments['N2_coded'] = (experiments['N2']-m3)/s3
experiments['N3_coded'] = (experiments['N3']-m4)/s4
experiments['Epoch_coded'] = (experiments['Epoch']-m5)/s5
experiments['Batch_coded'] = (experiments['Batch']-m6)/s6
experiments['Layer_coded'] = (experiments['Layer']-m7)/s7
print(experiments)        


        
#Tunning Part Begins_________________________________________________________________________________________________________________________________________________________________




optimizers = ['Adagrad', 'Adadelta', 'SGD', 'RMSprop', 'Adam', 'Adamax', 'Nadam']

for i in range(132):
    print(i)
    q1 = experiments.iloc[i, 1]             # number of neurons in first hidden layer
    q2 = experiments.iloc[i, 2]              # number of neurons in second hidden layer
    q3 = experiments.iloc[i, 3]             # number of neurons in third hidden layer
    epochs = experiments.iloc[i, 4]
    batch_size = experiments.iloc[i, 5]
    q4 = 15              # fixed number of neurons for each additional layer
    
    
    
    
    #GLM preprocessing begins------------------------------------------

    #Grouping ID
    distinct = data.drop(columns=['IDpol', 'Exposure', 'ClaimNb']).drop_duplicates().reset_index(drop=True)
    distinct['group_id'] = distinct.index + 1


    dat = data.merge(distinct, how='left')
    dat['ClaimNb'] = np.minimum(dat['ClaimNb'].astype(int), 4)
    dat['VehAge'] = np.minimum(dat['VehAge'], 20)
    dat['DrivAge'] = np.minimum(dat['DrivAge'], 90)
    dat['BonusMalus'] = np.minimum(dat['BonusMalus'], 150)
    dat['Density'] = np.round(np.log1p(dat['Density']), 2)
    dat['VehGas'] = pd.Categorical(dat['VehGas'])
    dat['Exposure'] = np.minimum(dat['Exposure'], 1)

    dat2 = dat.copy()  # Create a copy of the original DataFrame
    dat2['AreaGLM']=dat2['Area']
    dat2['AreaGLM'].replace({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6}, inplace=True)
    dat2['VehPowerGLM'] = pd.cut(dat2['VehPower'], bins=[-float('inf'), 9, float('inf')], labels=['9', '1-8'])
    dat2['VehAgeGLM'] = pd.cut(dat2['VehAge'], bins=[-float('inf'), 0, 10, float('inf')], labels=['1', '2', '3'])
    dat2['DrivAgeGLM'] = pd.cut(dat2['DrivAge'], bins=[-float('inf'), 20, 25, 30, 40, 50, 70, float('inf')], labels=['1', '2', '3', '4', '5', '6', '7'])
    dat2['BonusMalusGLM'] = dat2['BonusMalus'].clip(upper=150).astype(int)
    dat2['DensityGLM'] = dat2['Density'].apply(lambda x: round(np.log1p(x), 2))
    dat2['VehAgeGLM'] = dat2['VehAgeGLM'].cat.reorder_categories(['2', '1', '3'], ordered=True)
    dat2['DrivAgeGLM'] = dat2['DrivAgeGLM'].cat.reorder_categories(['5', '1', '2', '3', '4', '6', '7'], ordered=True)
    dat2['Region'] = dat2['Region'].cat.set_categories(['R24'] + [cat for cat in sorted(dat2['Region'].cat.categories) if cat != 'R24'], ordered=True)
    dat2['log_DrivAge'] = np.log1p(dat2['DrivAge'])
    dat2['DrivAge**2'] = dat2['DrivAge'] ** 2
    dat2['DrivAge**3'] = dat2['DrivAge'] ** 3
    dat2['DrivAge**4'] = dat2['DrivAge'] ** 4
    dat2 = pd.get_dummies(dat2, columns=['VehPowerGLM','VehAgeGLM','DrivAgeGLM' , 'VehBrand', 'VehGas', 'Region'], drop_first=True)

    # Convert boolean columns to integers
    bool_cols = dat2.select_dtypes(include=['bool']).columns
    dat2[bool_cols] = dat2[bool_cols].astype(int)

    #Partitioning of Test and Training Sets
    group_id = dat2['group_id']

    # Get unique group IDs
    unique_groups = group_id.unique()

    # Split unique groups into train and test sets
    train_groups, test_groups = train_test_split(unique_groups, train_size=0.8, test_size=0.2, random_state=seed)

    # Filter DataFrame based on train and test group IDs
    train = dat2[group_id.isin(train_groups)]
    test = dat2[group_id.isin(test_groups)]

    #GLM preprocessing ends--------------------------------


    # Initialize DataFrame to store model results
    df_cmp = pd.DataFrame(columns=['model', 'epochs', 'run_time', 'parameters', 'in_sample_loss', 'out_sample_loss', 'avg_freq'])

    # Start measuring execution time
    start_time = time.time()
    glm2 = sm.GLM(
        train['ClaimNb'],
        train[['AreaGLM', 'VehPowerGLM_1-8', 'VehAgeGLM_1', 
               'VehAgeGLM_3', 'BonusMalusGLM', 'VehBrand_B10', 
               'VehBrand_B11', 'VehBrand_B12', 'VehBrand_B13', 
               'VehBrand_B14', 'VehBrand_B2', 'VehBrand_B3', 
               'VehBrand_B4', 'VehBrand_B5', 'VehBrand_B6',  
               'VehGas_Regular', 'DensityGLM','Region_R11', 
               'Region_R21', 'Region_R22', 'Region_R23', 
               'Region_R25', 'Region_R26', 'Region_R31', 
               'Region_R41', 'Region_R42', 'Region_R43', 
               'Region_R52', 'Region_R53', 'Region_R54', 
               'Region_R72', 'Region_R73', 'Region_R74', 
               'Region_R82', 'Region_R83', 'Region_R91', 
               'Region_R93', 'Region_R94', 'DrivAgeGLM_1', 
               'DrivAgeGLM_2', 'DrivAgeGLM_3', 'DrivAgeGLM_4', 
               'DrivAgeGLM_6', 'DrivAgeGLM_7', 'DrivAge', 
               'log_DrivAge', 'DrivAge**2', 'DrivAge**3', 
               'DrivAge**4']],
        offset=np.log1p(train['Exposure']),
        family=sm.families.Poisson()
    ).fit()
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time


    #Loss calculation data processing begins---------------
    predictors = ['AreaGLM', 'VehPowerGLM_1-8', 'VehAgeGLM_1', 
           'VehAgeGLM_3', 'BonusMalusGLM', 'VehBrand_B10', 
           'VehBrand_B11', 'VehBrand_B12', 'VehBrand_B13', 
           'VehBrand_B14', 'VehBrand_B2', 'VehBrand_B3', 
           'VehBrand_B4', 'VehBrand_B5', 'VehBrand_B6',  
           'VehGas_Regular', 'DensityGLM','Region_R11', 
           'Region_R21', 'Region_R22', 'Region_R23', 
           'Region_R25', 'Region_R26', 'Region_R31', 
           'Region_R41', 'Region_R42', 'Region_R43', 
           'Region_R52', 'Region_R53', 'Region_R54', 
           'Region_R72', 'Region_R73', 'Region_R74', 
           'Region_R82', 'Region_R83', 'Region_R91', 
           'Region_R93', 'Region_R94', 'DrivAgeGLM_1', 
           'DrivAgeGLM_2', 'DrivAgeGLM_3', 'DrivAgeGLM_4', 
           'DrivAgeGLM_6', 'DrivAgeGLM_7', 'DrivAge', 
           'log_DrivAge', 'DrivAge**2', 'DrivAge**3', 
           'DrivAge**4']

    test_ClaimNb = test['ClaimNb']
    test_Exposure = test['Exposure']
    test_ID=test['IDpol']

    test = test[predictors]
    dat2 = dat2[predictors]

    # List of columns to keep
    columns_to_keep = ['AreaGLM', 'VehPowerGLM_1-8', 'VehAgeGLM_1', 
           'VehAgeGLM_3', 'BonusMalusGLM', 'VehBrand_B10', 
           'VehBrand_B11', 'VehBrand_B12', 'VehBrand_B13', 
           'VehBrand_B14', 'VehBrand_B2', 'VehBrand_B3', 
           'VehBrand_B4', 'VehBrand_B5', 'VehBrand_B6',  
           'VehGas_Regular', 'DensityGLM','Region_R11', 
           'Region_R21', 'Region_R22', 'Region_R23', 
           'Region_R25', 'Region_R26', 'Region_R31', 
           'Region_R41', 'Region_R42', 'Region_R43', 
           'Region_R52', 'Region_R53', 'Region_R54', 
           'Region_R72', 'Region_R73', 'Region_R74', 
           'Region_R82', 'Region_R83', 'Region_R91', 
           'Region_R93', 'Region_R94', 'DrivAgeGLM_1', 
           'DrivAgeGLM_2', 'DrivAgeGLM_3', 'DrivAgeGLM_4', 
           'DrivAgeGLM_6', 'DrivAgeGLM_7', 'DrivAge', 
           'log_DrivAge', 'DrivAge**2', 'DrivAge**3', 
           'DrivAge**4']




    # List of all columns
    all_columns = test.columns.tolist()

    # List of columns to drop
    columns_to_drop = [col for col in all_columns if col not in columns_to_keep]

    # Drop the columns
    df_subset = test.drop(columns=columns_to_drop)
    test=df_subset

    # List of all columns
    all_columns = dat2.columns.tolist()

    # List of columns to drop
    columns_to_drop = [col for col in all_columns if col not in columns_to_keep]

    # Drop the columns
    df_subset = dat2.drop(columns=columns_to_drop)
    dat2=df_subset

    # Predictions
    test['fitGLM2'] = glm2.predict(test)
    train['fitGLM2'] = glm2.fittedvalues
    dat2['fitGLM2'] = glm2.predict(dat2)

    test['ClaimNb']=test_ClaimNb
    test['Exposure']=test_Exposure

    #Loss Cacluation Data Processing ends--------------------


    # In-sample and out-of-sample losses
    train_deviance = poisson_deviance(train['fitGLM2'], train['ClaimNb'])
    test_deviance = poisson_deviance(test['fitGLM2'], test['ClaimNb'])


    # Overall estimated frequency
    average_frequency = sum(test['fitGLM2']) / sum(test['Exposure'])

    average_freq=sum(train['ClaimNb'])/sum(train['Exposure'])
    coef_glm2_length=len(glm2.params)

    new_row = pd.DataFrame({
        'model': ["M1: GLM"],
        'epochs': [np.nan],
        'run_time': [round(execution_time, 0)],
        'parameters': [coef_glm2_length],
        'in_sample_loss': [round(poisson_deviance(train['fitGLM2'], train['ClaimNb']), 4)],
        'out_sample_loss': [round(poisson_deviance(test['fitGLM2'], test['ClaimNb']), 4)],
        'avg_freq': [round(np.sum(test['fitGLM2']) / np.sum(test['Exposure']), 4)]
    })

    # Append the new row to df_cmp
    df_cmp = pd.concat([df_cmp, new_row], ignore_index=True)

    #print(df_cmp)
















    #CANN Data Preprocess Begins------------------------------

    test['IDpol']=test_ID
    dat['Area'].replace({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6}, inplace=True)
    dat['VehGas'].replace({'Diesel':1, 'Regular':2}, inplace=True)
    dat['VehBrandX']=dat['VehBrand']
    dat['VehBrandX'].replace({'B1':1, 'B2':2, 'B3':3, 'B4':4, 'B5':5, 'B6':6, 'B10':7, 'B11':8, 'B12':9, 'B13':10, 'B14':11}, inplace=True)
    dat['RegionX']=dat['Region']
    dat['RegionX'].replace({'R11':1, 'R21':2, 'R22':3, 'R23':4, 'R24':5, 'R25':6, 'R26':7, 'R31':8, 'R41':9, 'R42':10, 'R43':11, 'R52':12, 'R53':13, 'R54':14, 'R72':15, 'R73':16, 'R74':17, 'R82':18, 'R83':19, 'R91':20, 'R93':21, 'R94':22}, inplace=True)
    predictions = pd.concat([train[['IDpol', 'fitGLM2']], test[['IDpol', 'fitGLM2']]], axis=0, ignore_index=True)
    dat = dat.merge(predictions, on='IDpol', how='left')




    def preprocess_minmax(varData):
        X = np.asarray(varData, dtype=float)
        return 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

    def preprocess_catdummy(data, varName, prefix):
        # Convert the specified column to categorical if it's not already
        data[varName] = pd.Categorical(data[varName])
        
        # Create dummy variables, dropping the first to avoid multicollinearity
        dummies = pd.get_dummies(data[varName], prefix=prefix, drop_first=True)
        
        # Concatenate the original DataFrame with the dummy variables
        data_with_dummies = pd.concat([data, dummies], axis=1)
        
        return data_with_dummies

    def preprocess_features(data):
        # Applying MinMax scaling to specified columns
        for col in ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
            scaled_col = preprocess_minmax(data[col])
            data[f"{col}X"] = scaled_col
        
        # Adjusting values for specific columns
        data['VehGasX'] = data['VehGas'].astype(int) - 1.5
        data['VehBrandX'] = data['VehBrandX'].astype(int) - 1
        data['RegionX'] = data['RegionX'].astype(int) - 1

        # Applying dummy coding
        data = preprocess_catdummy(data, "VehBrand", "VehBrand")
        data = preprocess_catdummy(data, "Region", "Region")
        
        return data


    dat2 = preprocess_features(dat)
    # Convert boolean columns to integers
    bool_cols = dat2.select_dtypes(include=['bool']).columns
    dat2[bool_cols] = dat2[bool_cols].astype(int)




    dat2['group_id'] = dat2.index + 1

    #Partitioning of Test and Training Sets
    group_id = dat2['group_id']

    # Get unique group IDs
    unique_groups = group_id.unique()

    # Split unique groups into train and test sets
    train_groups, test_groups = train_test_split(unique_groups, train_size=0.8, test_size=0.2, random_state=seed)

    # Filter DataFrame based on train and test group IDs
    train = dat2[group_id.isin(train_groups)]
    test = dat2[group_id.isin(test_groups)]

    #CANN Data Preprocess Ends----------------------------------------
    
    

    # Claims frequency of train/test
    train_frequency = round(train['ClaimNb'].sum() / train['Exposure'].sum(), 4)
    test_frequency = round(test['ClaimNb'].sum() / test['Exposure'].sum(), 4)



    # Define features
    features = [
        "AreaX", "VehPowerX", "VehAgeX", "DrivAgeX", "BonusMalusX", "DensityX", "VehGasX",
        "VehBrand_B2", "VehBrand_B3", "VehBrand_B4", "VehBrand_B5", "VehBrand_B6", "VehBrand_B10", "VehBrand_B11", "VehBrand_B12", "VehBrand_B13", "VehBrand_B14",
        'Region_R21', 'Region_R22', 'Region_R23', 'Region_R24', 'Region_R25', 'Region_R26', 'Region_R31', 'Region_R41', 'Region_R42', 'Region_R43', 'Region_R52', 'Region_R53', 'Region_R54', 'Region_R72', 'Region_R73', 'Region_R74', 'Region_R82', 'Region_R83', 'Region_R91', 'Region_R93', 'Region_R94'
    ]

    Xtrain = train[features].to_numpy()
    Xtest = test[features].to_numpy()


    lambda_hom = np.sum(train['ClaimNb']) / np.sum(train['Exposure'])

    # Define the non-embedded features
    features = ["AreaX", "VehPowerX", "VehAgeX", "DrivAgeX", "BonusMalusX", "VehGasX", "DensityX"]
    q0 = len(features)   # number of non-embedded input features
    verbose = 1
    validation_split = 0.2





    # Training data
    Xtrain = train[features].to_numpy()  # design matrix training sample
    VehBrandtrain = train['VehBrandX'].to_numpy().reshape(-1, 1)
    Regiontrain = train['RegionX'].to_numpy().reshape(-1, 1)
    Ytrain = train['ClaimNb'].to_numpy().reshape(-1, 1)

    # Testing data
    Xtest = test[features].to_numpy()  # design matrix test sample
    VehBrandtest = test['VehBrandX'].to_numpy().reshape(-1, 1)
    Regiontest = test['RegionX'].to_numpy().reshape(-1, 1)
    Ytest = test['ClaimNb'].to_numpy().reshape(-1, 1)

    # Choosing the right volumes for CANN (GLM predictions to be fed through skip connection)
    Vtrain = np.log1p(train['fitGLM2'].values).reshape(-1, 1)
    Vtest = np.log1p(test['fitGLM2'].values).reshape(-1, 1)
    lambda_hom = np.sum(train['ClaimNb']) / np.sum(train['fitGLM2'])


    # Set the number of levels for the embedding variables
    VehBrandLabel = len(train['VehBrandX'].unique())
    RegionLabel = len(train['RegionX'].unique())


    # Dimensions embedding layers for categorical features
    d = 2        

    # Define the network architecture
    Design = Input(shape=(q0,), dtype='float32', name='Design')
    VehBrand = Input(shape=(1,), dtype='int32', name='VehBrand')
    Region = Input(shape=(1,), dtype='int32', name='Region')
    LogVol = Input(shape=(1,), dtype='float32', name='LogVol')

    BrandEmb = Embedding(input_dim=VehBrandLabel, output_dim=d, input_length=1, name='BrandEmb')(VehBrand)
    BrandEmb = Flatten(name='Brand_flat')(BrandEmb)

    RegionEmb = Embedding(input_dim=RegionLabel, output_dim=d, input_length=1, name='RegionEmb')(Region)
    RegionEmb = Flatten(name='Region_flat')(RegionEmb)

    Network = Concatenate(name='concate')([Design, BrandEmb, RegionEmb])
    Network = Dense(units=q1, activation='tanh', name='hidden1')(Network)
    Network = Dense(units=q2, activation='tanh', name='hidden2')(Network)
    
    if experiments.iloc[i, 6] == 2:
        print("K=2 Layer")
    elif experiments.iloc[i, 6] == 3:
        Network = Dense(units=q3, activation='tanh', name='hidden3')(Network)
        print("K=3 Layer")
    else:
        Network = Dense(units=q3, activation='tanh', name='hidden3')(Network)
        Network = Dense(units=q4, activation='tanh', name='hidden4')(Network)
        print("K=4 Layer")
    
    
    
    
    
    Network_output = Dense(units=1, activation='linear', name='Network')(Network)

    Response = Add()([Network_output, LogVol])
    Response = Dense(units=1, activation=tf.exp, name='Response', trainable=False)(Response)

    model_cann = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=Response)

    # Setting weights separately after defining the model
    
    if experiments.iloc[i, 6] == 2:
        print("K=2 Layer")
        model_cann.get_layer('Network').set_weights([np.zeros((q2, 1)), np.full((1,), np.log1p(lambda_hom))])

    elif experiments.iloc[i, 6] == 3:
        print("K=3 Layer")
        model_cann.get_layer('Network').set_weights([np.zeros((q3, 1)), np.full((1,), np.log1p(lambda_hom))])

    else:
        model_cann.get_layer('Network').set_weights([np.zeros((q4, 1)), np.full((1,), np.log1p(lambda_hom))])
        print("K=4 Layer")
    
    
    
    model_cann.get_layer('Response').set_weights([np.ones((1, 1)), np.zeros((1,))])

    model_cann.compile(
        loss='poisson',
        optimizer=optimizers[experiments.iloc[i, 0]]
    )
    

    model_cann.summary()

    # Fitting the neural network
    start_time = time.time()
    fit = model_cann.fit(
        [Xtrain, VehBrandtrain, Regiontrain, Vtrain], Ytrain,
        epochs=epochs, 
        batch_size=batch_size,
        verbose=verbose,
        validation_split=validation_split
    )
    exec_time = time.time() - start_time

    #plot_loss(fit)


    # Calculating the predictions
    train.loc[:, 'fitCANN'] = model_cann.predict([Xtrain, VehBrandtrain, Regiontrain, Vtrain]).flatten()
    test.loc[:, 'fitCANN'] = model_cann.predict([Xtest, VehBrandtest, Regiontest, Vtest]).flatten()



    # Calculating Poisson deviance for train and test sets
    train_poisson_deviance = poisson_deviance(train['fitCANN'], train['ClaimNb'])
    test_poisson_deviance = poisson_deviance(test['fitCANN'], test['ClaimNb'])



    # Average frequency
    average_frequency = np.sum(test['fitCANN']) / np.sum(test['Exposure'])


    # Function to count trainable parameters
    def count_trainable_params(model):
        return np.sum([np.prod(v.shape) for v in model.trainable_weights])

    # Calculate trainable parameters
    trainable_params = count_trainable_params(model_cann)

    try:
        df_cmp = df_cmp
    except NameError:
        df_cmp = pd.DataFrame()


    # Append the new row with model performance data
    new_row = pd.DataFrame({
        'model': ['M4: EmbCANN'],
        'epochs': [epochs],
        'run_time': [round(exec_time, 0)],
        'parameters': [trainable_params],
        'in_sample_loss': [round(poisson_deviance(train['fitCANN'], train['ClaimNb']), 4)],
        'out_sample_loss': [round(poisson_deviance(test['fitCANN'], test['ClaimNb']), 4)],
        'avg_freq': [round(np.sum(test['fitCANN']) / np.sum(test['Exposure']), 4)]
    })

    df_cmp = pd.concat([df_cmp, new_row], ignore_index=True)

    # Display the updated DataFrame
    print(df_cmp)
    
    
    
    experiments.loc[i, 'loss'] = test_poisson_deviance
    
    
#Saving the resulting losses
experiments.to_csv('experiments.csv', index=False)

    #_________________________________________________________________________________________________________________________________________________________________



















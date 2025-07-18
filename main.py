from function.test_gpu import check_gpu
from function.preprocess import preprocess
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
import keras_tuner as kt
from function.hypermodel import MyHyperModel
import io
from contextlib import redirect_stdout


#####Check gpu use####
check_gpu()

#####Preprocess data####
dataset_train = "Dataset/data_for_train.csv"
dataset_test = "Dataset/data_for_test.csv"
dataframe_train = pd.read_csv(dataset_train)
dataframe_test = pd.read_csv(dataset_test)

columns = ['EDA_Phasic', 'SCR_Amplitude', 'NumPeaks', 'HRV_RMSSD', 'HRV_LFHF',
            'HRV_SD1SD2', 'HRV_DFA_alpha1', 'HRV_SampEn','EDA_Tonic',
            'HRV_ApEn','SCR_Onsets','PPG_Quality', 'HR', 'gender', 'bmi','sleep','type','stress', 'id']

num_features = len(columns)

preprocess_train = preprocess(dataframe_train, columns)
dataframe_train = preprocess_train.select_columns()

preprocess_test = preprocess(dataframe_test, columns)
dataframe_test = preprocess_test.select_columns()
# print(dataframe.head())
# print(dataframe.tail())

print(dataframe_train['stress'].value_counts())

dataframe_train, mapping = preprocess_train.label_encoding()
dataframe_test, _ = preprocess_test.label_encoding()
print(dataframe_train.head())
print(mapping)

sc = StandardScaler()
# dataframe_train = preprocess.scale_data(sc, ['stress', 'id'])
# dataframe_test = preprocess.scale_data(sc, ['stress', 'id'])

dataframe_train = sc.fit_transform(dataframe_train)
dataframe_test = sc.transform(dataframe_test)

# print(dataframe['gender'].value_counts())
# dataframe.to_csv('Dataset/combined_data_preprocessed.csv', index=False)

#######Create Sequence###############
X_train, y_train = preprocess_train.create_sequence()
X_test, y_test = preprocess_test.create_sequence()

print(X_train.shape)
print(y_train.shape)

##########Smoote Data#########

# Flatten X
X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape Back
X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])

print(X_resampled.shape)
print(y_resampled.shape)

X_train = X_resampled
y_train = y_resampled


print(X_test.shape)
print(y_test.shape)


#########Split train test ##########
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"X train shape : {X_train.shape}")
# print(f"y train shape : {y_train.shape}")

######### Create Model and Tune Hyperparameter########

tuner = kt.BayesianOptimization(
    MyHyperModel(num_features=num_features),
    objective="val_accuracy",        # หรือเปลี่ยนเป็น metric อื่นที่อยาก optimize
    max_trials=100,                 # กี่ combinations ของ hyperparameter ที่จะลอง
    # executions_per_trial=1,           # วิ่งแต่ละครั้งกี่รอบ (สำหรับลด randomness)
    directory='my_tuner_results',
    project_name='stress_rnn',
)

tuner.search(
    X_train, y_train, # Use the sequenced data here
    X_test=X_test, y_test=y_test,
)


print(tuner.results_summary(num_trials=100))

f = io.StringIO()
with redirect_stdout(f):
    tuner.results_summary()
summary = f.getvalue()

with open('tuner_results_summary.txt', 'w') as file:
    file.write(summary)


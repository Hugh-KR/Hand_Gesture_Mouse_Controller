import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/shinchanghyeok/PycharmProjects/pythonProject/traveling_recommendation/REC_MODEL/Data/추천모델 학습 데이터 (1).csv')

# 데이터 복사
tmp = data.copy()

print(tmp.head())

# 데이터 전처리
feeling = tmp['feeling'].unique()
place = data['place'].unique()
von = data['von'].unique()

tmp=tmp.dropna()

feeling = tmp['feeling'].unique()
place = tmp['place'].unique()
von = tmp['von'].unique()

print(feeling)
print(place)
print(von)

tmp=tmp.drop(columns=['place'])

print(tmp.head())

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
x=encoder.fit_transform(tmp['feeling'])
tmp['feeling']=x

tmp=tmp.drop(columns=['Index'])
y_label=tmp['von']

tmp=tmp.drop(columns=['von'])

print(tmp.head())

from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
tmp_rs=rs.fit_transform(tmp)
print(len(tmp_rs))

print(len(y_label))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tmp_rs, y_label, test_size=0.2, random_state=42)

X_train_db=pd.DataFrame(X_train)

model = models.Sequential()
model.add(layers.Dense(units=512, activation='relu', input_shape=(22,)))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dense(units=8, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

results = model.fit(x=tmp_rs, y=y_label, batch_size=512, epochs=30, validation_split=0.2)


print(results.history.keys())

print(results.history['val_accuracy'])
from keras.models import load_model


model.summary()
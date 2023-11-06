# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:16:04 2023

@author: joaom
"""
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../dados_versao_final/doencas/MG.csv')
df = df.drop(0)
df2 = pd.read_csv('../dados_versao_final/meteorologicos/MG.csv',index_col='data',parse_dates=True)
df2 = df2.drop('2023-01-01')
df.index = df2.index



import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(10,6))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(df.index,df['casos'])
ax.set_ylabel('Casos')
ax.set_title('Casos de dengue relatados, por semana')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# Fonte https://colab.research.google.com/drive/1HxPsJvEAH8L7XTmLnfdJ3UQx7j0o1yX5?usp=sharing#scrollTo=AmtFJ_0pD-xh

#==============================================================================
# UNIVARIATE
#==============================================================================

# FUNCAO QUE VAI CONSTRUIR A JANELA DESLIZANTE
def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

ALVO = df['casos']

WINDOW_SIZE = 26 # HIPERPARAMETRO
X1, y1 = df_to_X_y(ALVO, WINDOW_SIZE)
#X1.shape, y1.shape

# SEPARACAO EM TREINO VALIDACAO E TESTE
X_train1, y_train1 = X1[:574], y1[:574]
X_val1, y_val1 = X1[574:626], y1[574:626]
X_test1, y_test1 = X1[626:], y1[626:]
#X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# MONTAGEM DO MODELO
model1 = Sequential()
model1.add(InputLayer((26, 1))) # Formato de X_train1
model1.add(LSTM(units=64)) # Rever esse numero -> neurons da camada
model1.add(Dense(8, 'relu')) # 8 tbm eh neuron de camada
model1.add(Dense(1, 'linear'))
model1.summary() # Mostra o formato do modelo, bonitinho

cp1 = ModelCheckpoint('./model1',save_best_only=True) # Primeiro parametro aqui eh a filepath do modelo
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# HIPERPARAMETROS
# Fitando no modelo
model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=200, callbacks=[cp1])
# Grafico da loss
plt.plot(range(len(model1.history.history['loss'])), model1.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()

# Carregar o melhor modelo q ele salvou
from tensorflow.keras.models import load_model
model1 = load_model('model1/')

# Dataframe comparativo
train_predictions = model1.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
# Graficos comparativos
import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:100]) # Rever aqui -> da um zoom pra nao ficar feio o grafico
plt.plot(train_results['Actuals'][50:100])

val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results
plt.plot(val_results['Val Predictions'][:100])
plt.plot(val_results['Actuals'][:100])

test_predictions = model1.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results
plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])

#==============================================================================
# MAIS UNIVARIATE
#==============================================================================

from sklearn.metrics import mean_squared_error as mse

# Funcao bonita pra plotar os previstos com os reais
def plot_predictions1(model, X, y, start=0, end=100):
  predictions = model.predict(X).flatten()
  df = pd.DataFrame(data={'Predictions': predictions, 'Actuals':y})
  plt.plot(df['Predictions'][start:end])
  plt.plot(df['Actuals'][start:end])
  return df, mse(predictions, y)
plot_predictions1(model1, X_test1, y_test1) # Chamada assim

# MONTAGEM DO MODELO
model2 = Sequential()
model2.add(InputLayer((26, 1))) # Rever esse numero
model2.add(Conv1D(64, kernel_size=2, activation='relu')) # kernel_size rever
model2.add(Flatten())
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))
model2.summary()

# Salvamento e compilacao, igual ao de univariate
cp2 = ModelCheckpoint('./model2', save_best_only=True)
model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# HIPERPARAMETRO epochs
model2.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, callbacks=[cp2])

plt.plot(range(len(model2.history.history['loss'])), model2.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()

plot_predictions1(model2, X_test1, y_test1, end=678)


#==============================================================================
# MULTIVARIATE
#==============================================================================



def df_to_X_y2(df, window_size=4): # HIPERPARAMETRO
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][0] # Rever aqui -> supoe o alvo estar na posicao 0
    y.append(label)
  return np.array(X), np.array(y)

# Inserindo a var alvo no mesmo df que as demais
df3 = df2.copy()
df3.insert(0,'casos',df['casos'])

X2, y2 = df_to_X_y2(df3) # Rever aqui para o df que esta sendo usado
#X2.shape, y2.shape



# Conjuntos de treino, teste e validacao
# Sugestao de usar 70-15-15
train_length = int(len(X2)*0.7)
test_length = int(len(X2)*0.15)
X2_train, y2_train = X2[:train_length], y2[:train_length] 
X2_val, y2_val = X2[train_length:train_length+test_length], y2[train_length:train_length+test_length]
X2_test, y2_test = X2[train_length+test_length:], y2[train_length+test_length:]


# Conjuntos de treino, teste e validacao
# Sugestao de usar 70-15-15 ->
#X2_train, y2_train = X2[:60000], y2[:60000] 
#X2_val, y2_val = X2[60000:65000], y2[60000:65000]
#X2_test, y2_test = X2[65000:], y2[65000:]
#X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape






# Pre processamento -> Standardization
# O cara faz o pre-processamento depois de dividir os conjuntos, vou fazer antes
temp_training_mean = np.mean(X2_train[:, :, 0])
temp_training_std = np.std(X2_train[:, :, 0])
def preprocess(X):
  X[:, :, 0] = (X[:, :, 0] - temp_training_mean) / temp_training_std 
  return X
# Chamando o preprocessamento
preprocess(X2_train)
preprocess(X2_val)
preprocess(X2_test)


model4 = Sequential()
model4.add(InputLayer((26, 7))) # 26 WS, 7 variaveis
model4.add(LSTM(64)) # HIPERPARAMETROS
model4.add(Dense(8, 'relu'))
model4.add(Dense(1, 'linear'))
model4.summary()

cp4 = ModelCheckpoint('./model4', save_best_only=True)
model4.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# HIPERPARAMETROS
model4.fit(X2_train, y2_train, validation_data=(X2_val, y2_val), epochs=10, callbacks=[cp4])

plot_predictions1(model4, X2_test, y2_test)





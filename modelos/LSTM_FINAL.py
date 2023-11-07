# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_m = pd.read_csv('../dados_versao_final/meteorologicos/MG.csv',index_col='data',parse_dates=True)
df_d = pd.read_csv('../dados_versao_final/doencas/MG.csv')

# Separacao variaveis
X = df_m[:].values
y = df_d['casos'].values

# Transformacao MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Montagem para o formato do modelo
WS = 26  # HIPERPARAMETRO

X = np.array([X[i-WS:i] for i in range(WS, len(X))]) # Original range(WS, len(X))
y = y[WS:]

# Separacao em treino/teste
#train_size = int(0.8 * len(X))
train_size = 600
X_train, X_test = X[:train_size-52], X[train_size:]
y_train, y_test = y[:train_size-52], y[train_size:]

# Desfaz transformacoes
def inverseTransform(data):
    return scaler.inverse_transform(data)

#=================================================================================
# INICIALIZANDO MODELO
#=================================================================================
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

# Camadas do modelo segundo artigo
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

def createModel(units, dropout):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units = units, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(units = units, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(units = units, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(units = units))
    model.add(Dropout(dropout))
    model.add(Dense(units = 1))
    return model

# Caso queira salvar/carregar modelo
from keras.models import load_model
def saveModel(model,path='./LSTM_final'):
    model.save(path)
def loadModel(path='./LSTM_final'):
    return load_model('./LSTM_final')

#=================================================================================
# SEM RODAR O GRIDSEARCH
#=================================================================================
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from scikeras.wrappers import KerasRegressor

def runModel_1(model,epochs,batch_size):
    # optimizer=RMSprop(learning_rate=0.001)
    model.compile(optimizer=Adam(learning_rate=0.001), loss = 'mean_squared_error',metrics=["accuracy", "mape", "mse"])
    model.fit(X_train,y_train, epochs = epochs, batch_size = batch_size)
    return model, model.predict(X_test) # Se vier esquisito, usar .flatten()

def runModel_2(model,epochs,batch_size):
    best_model = KerasRegressor(
        model=model,
 	    warm_start=False,
 	    random_state=None,
 	    optimizer=RMSprop(learning_rate=0.001),
     	loss='mean_squared_error',
 	    metrics=None,
 	    batch_size=64,
 	    validation_batch_size=None,
 	    verbose=True,
 	    callbacks=None,
 	    validation_split=0.0,
 	    shuffle=True,
 	    run_eagerly=False,
 	    epochs=200
    )
    # Consegui melhores resultados com epochs=500, batch_size=32, WS=26
    best_model.fit(X_train,y_train, epochs = epochs, batch_size = batch_size)
    return model, best_model.predict(X_test)

#=================================================================================
# COM GRID SEARCH
#=================================================================================
# com learning_rate=0.0001, ws = 26
# {'batch_size': 64, 'epochs': 200}
# best_model:
#     KerasRegressor(
# 	model=None
# 	build_fn=<keras.src.engine.sequential.Sequential object at 0x798858ed32e0>
# 	warm_start=False
# 	random_state=None
# 	optimizer=rmsprop
# 	loss=None
# 	metrics=None
# 	batch_size=64
# 	validation_batch_size=None
# 	verbose=True
# 	callbacks=None
# 	validation_split=0.0
# 	shuffle=True
# 	run_eagerly=False
# 	epochs=200)

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
def runModel_3(model):
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Grid search de hiperparametros
    lstm_model = KerasRegressor(build_fn=model, verbose=True, epochs=100)
    param_grid = {
        'units' : [32, 64, 128],
        'dropout_rate' : [0.2, 0.4, 0.6],
        'batch_size': [16, 32, 64],
        #'epochs': [100, 200, 300],
        # learning_rate: 0.0001 
        }
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    grid_search = GridSearchCV(
        #temp_folder = './temp/',
        n_jobs=-1,
        estimator=lstm_model, 
        param_grid=param_grid, 
        scoring='neg_mean_squared_error', # testar 'mean_squared_error', 
        verbose=2,
        cv=3
        )
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model, best_model.predict(X_test)

#=================================================================================
# AVALIACAO 
#=================================================================================

# Acredito que use as metricas do grid_search
# mse = best_model.evaluate(X_test, y_test, verbose=2)
# mse = model.evaluate(X_test, y_test, verbose=2)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
def printMetrics():
    print('R2 = ',r2_score(y_test, y_pred))
    print('MSE = ',mean_squared_error(y_test, y_pred))

#=================================================================================
# GRAFICOS
#=================================================================================

# Casos total
import matplotlib.dates as mdates
def fullGraph(pred=False,y_pred=[]):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.plot(df_m.index,df_d['casos'])
    if pred:
        ax.plot(df_m.index[-y_pred.shape[0]:], y_pred, label='Predicted Cases')
    ax.plot(df_m.index,y_pred)
    ax.set_ylabel('Casos')
    ax.set_title('Casos de dengue relatados, por semana')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Loss function - apos o metodo fit
def lossGraph(model):
    plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.show()

# Comparacao final
def compareGraph(y_pred):
    plt.plot(y_test, color = 'red', label = 'Numeros reais')
    plt.plot(y_pred, color = 'blue', label = 'Numeros previstos')
    plt.title('Previsao de casos - MG')
    plt.xlabel('Semanas')
    plt.ylabel('Nro de casos')
    plt.legend()
    plt.show()
# Grafico do todo - com as previsões inclusas
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(10,6))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(df_m.index,df_d['casos'])
ax.plot(df_m.index[-52:], y_pred, label='Predicted Cases')
ax.plot(df_m.index,y_pred)
ax.set_ylabel('Casos')
ax.set_title('Casos de dengue relatados, por semana')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()


#=================================================================================
# EXECUCAO
#=================================================================================

fullGraph()

model = createModel(units=64,dropout=0.2)
# saveModel(model)
# model = loadModel()

epochs = 500
batch_size=64
model, y_pred = runModel_1(model,epochs,batch_size)
# lossGraph(model)
# model, y_pred = runModel_2(model,epochs,batch_size)
# model, y_pred = runModel_3(model)

print(y_pred.shape[0])

compareGraph(y_pred)
y_pred = inverseTransform(y_pred)
y_test = inverseTransform(y_test)
compareGraph(y_pred)

fullGraph(pred=True,y_pred=y_pred)

printMetrics()







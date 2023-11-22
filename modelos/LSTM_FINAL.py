# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def openAndSplit(uf):
    df_m = pd.read_csv('../dados_versao_final/meteorologicos/'+uf+'.csv',index_col='data',parse_dates=True)
    df_d = pd.read_csv('../dados_versao_final/doencas/'+uf+'.csv')

    # Separacao variaveis
    X = df_m[:].values
    y = df_d['casos'].values
    
    return df_m, df_d, X, y

# Transformacao MinMax

def scaleData(X,y,scaler):    
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y

def prepareData(WS,train_size,X,y):
    # Montagem para o formato do modelo
    X = np.array([X[i-WS:i] for i in range(WS, len(X))]) # Original range(WS, len(X))
    y = y[WS:]

    # Separacao em treino/teste
    X_train, X_test = X[:train_size-52], X[train_size:]
    y_train, y_test = y[:train_size-52], y[train_size:]
    
    return X_train, X_test, y_train, y_test
    
# Desfaz transformacoes
def inverseTransform(data,scaler):
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

def createModel(units, dropout, X_train):
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


# Early stopping ou patience
# Justificar a escolha de 26 semanas
# Hiper de poda de arvores, profundidade maxima
# biblioteca XGBoost
# primeiro testar os tres sem google, depois com; comparacoes a cada vez.
# logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
#                                  scoring='roc_auc', n_jobs=1, cv=time_split, verbose=1)
# time_split = TimeSeriesSplit(n_splits=10)
# Trazer tabelas e graficos
# Overleaf modelo 


from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from keras.callbacks import EarlyStopping

#from scikeras.wrappers import KerasRegressor
def runModelGridSearch(model, X_train):
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Grid search de hiperparametros
    time_series = TimeSeriesSplit(n_splits=5)    
    lstm_model = KerasRegressor(build_fn=model, verbose=True, epochs=300)
    param_grid = {
        #'units' : [32, 64, 128],
        #'dropout_rate' : [0.2, 0.4, 0.6],
        'batch_size': [30, 60, 100],
        'epochs': [300],
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
        cv=time_series.split(X_train)
        )
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    grid_search.fit(X_train, y_train,callbacks=[early_stopping])
    print(grid_search.best_params_) # {'batch_size': 100, 'epochs': 300}
    best_model = grid_search.best_estimator_
#     KerasRegressor(
# 	model=None
# 	build_fn=<keras.src.engine.sequential.Sequential object at 0x7f4f48adfa60>
# 	warm_start=False
# 	random_state=None
# 	optimizer=rmsprop
# 	loss=None
# 	metrics=None
# 	batch_size=100
# 	validation_batch_size=None
# 	verbose=True
# 	callbacks=None
# 	validation_split=0.0
# 	shuffle=True
# 	run_eagerly=False
# 	epochs=300
# )
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

# from sklearn.metrics import *
# def calculate_metrics(pred, ytrue):
#     return [mean_absolute_error(ytrue, pred, ), explained_variance_score(ytrue, pred),
#             mean_squared_error(ytrue, pred), mean_squared_log_error(ytrue, pred),
#             median_absolute_error(ytrue, pred), r2_score(ytrue, pred)]

#=================================================================================
# GRAFICOS
#=================================================================================

# Casos total
import matplotlib.dates as mdates
def fullGraph(pred=False,y_pred=[],save=False,uf = ''):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.plot(df_m.index,df_d['casos'])
    if pred:
        ax.plot(df_m.index[-y_pred.shape[0]:], y_pred, label='Predicted Cases')
    #ax.plot(df_m.index,y_pred)
    ax.set_ylabel('Casos')
    ax.set_title('Casos de dengue relatados, por semana')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(fname = uf+'.png',format='png')

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

#=================================================================================
# EXECUCAO
#=================================================================================

# 14 : 'RR',
UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 15 : 'PA', 16 : 'AP', 17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}
WS = 26
train_size = 600
scaler = MinMaxScaler()
epochs = 300
batch_size=100
for uf in UFCodes.values():
    df_m, df_d, X, y = openAndSplit(uf)
    X, y = scaleData(X, y, scaler)
    X_train, X_test, y_train, y_test = prepareData(WS, train_size, X, y)
    model = createModel(units=64,dropout=0.2,X_train=X_train)
    model, y_pred = runModel_1(model,epochs,batch_size)
    y_pred = inverseTransform(y_pred,scaler)
    y_test = inverseTransform(y_test,scaler)
    fullGraph(pred=True,y_pred=y_pred,save=True,uf=uf)


# PARA UM SÓ
df_m, df_d, X, y = openAndSplit('MG')
scaler = MinMaxScaler()

df_m = df_m.drop('2023-01-01')

df_m.to_csv('../dados_versao_final/meteorologicos/RS.csv')

X, y = scaleData(X, y, scaler)

fullGraph()

WS = 26
train_size = 600
#train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test =  prepareData(WS, train_size, X, y)

model = createModel(units=64,dropout=0.2,X_train=X_train)
# saveModel(model)
# model = loadModel()

epochs = 300
batch_size=100
model, y_pred = runModel_1(model,epochs,batch_size)
# lossGraph(model)
model, y_pred = runModelGridSearch(model, X_train)



print(y_pred.shape[0])

compareGraph(y_pred)
y_pred = inverseTransform(y_pred,scaler)
y_test = inverseTransform(y_test,scaler)
compareGraph(y_pred)

fullGraph(pred=True,y_pred=y_pred)

printMetrics()



model.save('./LSTM_keras/')





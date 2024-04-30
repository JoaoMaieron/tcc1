# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Abre os csv e separa em x e y - sem os trends ainda
def openAndSplit(uf):
    df_m = pd.read_csv('../dados_versao_final/meteorologicos/'+uf+'.csv',index_col='data',parse_dates=True)
    df_d = pd.read_csv('../dados_versao_final/doencas/'+uf+'.csv')

    # Separacao variaveis
    X = df_m[:].values
    y = df_d['casos'].values
    
    return df_m, df_d, X, y

# Aplica transformacao MinMax
def scaleData(X,y,scaler):    
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y

# Monta o conjunto de treino para o formato que o LSTM aceita como entrada
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
# Aqui é onde tem que fazer ainda mais alterações - quantidade de camadas, neuronios
def createModel(units, dropout, X_train):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units = 128, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(units = 64, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(units = 512, return_sequences = True))
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
# from tensorflow.keras.optimizers import RMSprop
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping

# Compile e fit do modelo
def runModel_1(model,epochs,batch_size,X_train,y_train,X_test):
    # optimizer=RMSprop(learning_rate=0.001)
    early_stopping = EarlyStopping(monitor='mse', patience=50, restore_best_weights=True)
    model.compile(optimizer=Adam(learning_rate=0.001), loss = 'mean_squared_error',metrics=["accuracy", "mape", "mse"])
    
    model.fit(X_train,y_train, epochs = epochs, batch_size = batch_size,callbacks=[early_stopping])
    return model, model.predict(X_test) # Se vier esquisito, usar .flatten()

#=================================================================================
# COM GRID SEARCH
#=================================================================================
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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

    return best_model, best_model.predict(X_test)

#=================================================================================
# AVALIACAO 
#=================================================================================
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
def fullGraph(df_m,df_d,pred=False,y_pred=[],save=False,uf = ''):
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
# Rodando todos os arqs e seja o que deus quiser
import csv
def runAll():
    # RR ta faltando coisa, AP nao faz predicao
    # 14 : 'RR', 16 : 'AP',
    UFCodes = {
        11 : 'RO', 12 : 'AC', 13 : 'AM', 15 : 'PA',  17 : 'TO', 
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
    results = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = openAndSplit(uf)
        X, y = scaleData(X, y, scaler)
        X_train, X_test, y_train, y_test = prepareData(WS, train_size, X, y)
        model = createModel(units=64,dropout=0.2,X_train=X_train)
        model, y_pred = runModel_1(model,epochs,batch_size,X_train,y_train,X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        y_pred = inverseTransform(y_pred,scaler)
        y_test = inverseTransform(y_test,scaler)
        fullGraph(df_m,df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        results.append({'estado': uf, 'R2': r2, 'MSE': mse})
    with open('metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

runAll()

# PARA UM SÓ
df_m, df_d, X, y = openAndSplit('DF')
scaler = MinMaxScaler()

df_d = df_d.drop(0)
df_d.to_csv('../dados_versao_final/doencas/DF.csv',index=False)

df_m = df_m.drop('2023-01-01')

df_m.to_csv('../dados_versao_final/meteorologicos/DF.csv')

X, y = scaleData(X, y, scaler)

fullGraph()

WS = 26
train_size = 600
#train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test =  prepareData(WS, train_size, X, y)

model = createModel(units=64,dropout=0.2,X_train=X_train)
# saveModel(model)
# model = loadModel()

epochs = 10
batch_size=100
model, y_pred = runModel_1(model,epochs,batch_size,X_train,y_train,X_test)
# lossGraph(model)
model, y_pred = runModelGridSearch(model, X_train)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(y_pred.shape[0])

compareGraph(y_pred)
y_pred = inverseTransform(y_pred,scaler)
y_test = inverseTransform(y_test,scaler)
compareGraph(y_pred)

fullGraph(pred=True,y_pred=y_pred)

printMetrics()



model.save('./LSTM_keras/')





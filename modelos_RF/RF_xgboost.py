# -*- coding: utf-8 -*-
import xgboost
import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 15 : 'PA',  17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}

# print(xgboost.__version__)

# https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

def openAndSplit(uf):
    df_m = pd.read_csv('../dados_versao_final/meteorologicos/'+uf+'.csv',index_col='data',parse_dates=True)
    df_d = pd.read_csv('../dados_versao_final/doencas/'+uf+'.csv')

    # No XGBoost precisa estar a variável alvo junto no conju
    y = df_d['casos'].values
    X = df_m[:].values
    #X = np.append(X, y.reshape(len(y),1), axis = 1)
    
    return df_m, df_d, X, y

# Aplica transformacao MinMax
def scaleData(X,y,scaler):    
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y

# Desfaz transformacoes
def inverseTransform(data,scaler):
    return scaler.inverse_transform(data)


#=================================================================================
# GRAFICOS
#=================================================================================

# Casos total
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


#=================================================================================
# TUNING HIPERPARAMETROS
#=================================================================================



def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
        'max_depth' : trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90) # usar suggest_float
        }
    tscv = TimeSeriesSplit(n_splits=5)
    eval_results = []
    for train_index, test_index in tscv.split(X_train):
        X_train_split, X_val = X_train[train_index], X_train[test_index]
        y_train_split, y_val = y_train[train_index], y_train[test_index]
        
        model = xgboost.XGBRFRegressor(**params, xgb_model=None)
        
        model.fit(X_train_split, y_train_split)
        
        y_pred = model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        
        eval_results.append(mse)
        
        average_mse = np.mean(eval_results)

    return average_mse # Pode-se alterar a metrica de referencia aqui

def runAllWithTuning():
    
    study = optuna.create_study(direction='minimize')
    
    for uf in UFCodes.values():
    
        df_m, df_d, X, y = openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = scaleData(X, y, scaler)
        X_train, X_test, y_train, y_test = X[:600], X[600:], y[:600], y[600:]
        
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
    
        best_params = study.best_params
        final_model = xgboost.XGBRFRegressor(**best_params, xgb_model=None)
        final_model.fit(X_train, y_train)
    
        y_pred = final_model.predict(X_test)
        y_pred = inverseTransform(y_pred.reshape(-1,1),scaler)
    
        fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)

runAllWithTuning()


#=================================================================================
# RODANDO
#=================================================================================

import csv
def runAll():
    # RR ta faltando coisa, AP nao faz predicao
    # 14 : 'RR', 16 : 'AP',        
    train_size = 600
    scaler = MinMaxScaler()
    
    #results = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = openAndSplit(uf)
        X, y = scaleData(X, y, scaler)
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]        
        model = xgboost.XGBRFRegressor(n_samples=1000)
        
        tscv = TimeSeriesSplit(n_splits=5)
        eval_results = []
        for train_index, test_index in tscv.split(X_train):
            X_train_split, X_val = X_train[train_index], X_train[test_index]
            y_train_split, y_val = y_train[train_index], y_train[test_index]
            # Train the model
            model.fit(X_train_split, y_train_split)

            # Make predictions on the validation set
            y_pred = model.predict(X_val)
            # Evaluate the model
            mse = mean_squared_error(y_val, y_pred)
            eval_results.append(mse)
            
        average_mse = np.mean(eval_results)
        y_pred = model.predict(X_test)
        y_pred = inverseTransform(y_pred.reshape(-1,1),scaler)

        fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        

runAll()


# subsample é a qtde de instancias por arvore acho -> default 0.8
model = xgboost.XGBRFRegressor(n_samples=1000)

df_m, df_d, X, y = openAndSplit('SC')

scaler = MinMaxScaler()
X, y = scaleData(X, y, scaler)

X_train, X_test, y_train, y_test = X[:600], X[600:], y[:600], y[600:]

tscv = TimeSeriesSplit(n_splits=5)
eval_results = []
for train_index, test_index in tscv.split(X_train):
    X_train_split, X_val = X_train[train_index], X_train[test_index]
    y_train_split, y_val = y_train[train_index], y_train[test_index]

    # Train the model
    model.fit(X_train_split, y_train_split)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Evaluate the model
    mse = mean_squared_error(y_val, y_pred)
    
    # Store evaluation result
    eval_results.append(mse)

average_mse = np.mean(eval_results)

y_pred = model.predict(X_test)

print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test,y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_percentage_error(y_test, y_pred))

y_pred = inverseTransform(y_pred.reshape(-1,1),scaler)


fullGraph(df_m, df_d,pred=True,y_pred=y_pred)

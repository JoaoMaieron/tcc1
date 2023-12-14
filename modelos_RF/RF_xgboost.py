import xgboost
import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
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
    ax.plot(df_m.index,df_d['casos'], label='Reais')
    if pred:
        ax.plot(df_m.index[-y_pred.shape[0]:], y_pred, label='Previstos')
    #ax.plot(df_m.index,y_pred)
    ax.set_ylabel('Casos')
    ax.set_title('Casos por semana - '+uf)
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

# https://forecastegy.com/posts/multiple-time-series-forecasting-with-scikit-learn/
# https://forecastegy.com/posts/multiple-time-series-forecasting-with-xgboost-in-python/#how-to-prepare-time-series-data-for-xgboost-in-python
def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
        'max_depth' : trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.50, 0.90),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1)
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
        
    return np.mean(eval_results) # Pode-se alterar a metrica de referencia aqui

import csv
def runAllWithTuning(train_size):
    
    study = optuna.create_study(direction='minimize')
    
    results = []
    scaler = MinMaxScaler()
    for uf in UFCodes.values():
        # Preparo dos dados
        df_m, df_d, X, y = openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = scaleData(X, y, scaler)
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        
        # Roda tuning
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
        # Obtem resultados
        best_params = study.best_params
        final_model = xgboost.XGBRFRegressor(**best_params, xgb_model=None)
        
        # Previsoes
        final_model.fit(X_train, y_train)    
        y_pred = final_model.predict(X_test)
        
        # Metricas
        r2 = round(r2_score(y_test, y_pred),3)
        mse = round(mean_squared_error(y_test, y_pred),3)
        mape = round(mean_absolute_percentage_error(y_test, y_pred),3)
        
        # Printa grafico
        y_pred = inverseTransform(y_pred.reshape(-1,1),scaler)    
        fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parametros':best_params})
    
    # Salva metricas em txt
    with open('metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAPE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

# runAllWithTuning(train_size=626)

#=================================================================================
# RODANDO SEM TUNING
#=================================================================================

# # import csv
# def runAll(train_size):
#     # RR ta faltando coisa, AP nao faz predicao
#     # 14 : 'RR', 16 : 'AP',        
#     scaler = MinMaxScaler()
    
#     results = []
#     for uf in UFCodes.values():
#         df_m, df_d, X, y = openAndSplit(uf)
#         X, y = scaleData(X, y, scaler)
#         X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]        
#         model = xgboost.XGBRFRegressor(n_samples=1000)
        
#         tscv = TimeSeriesSplit(n_splits=5)
#         # eval_results = []
#         for train_index, test_index in tscv.split(X_train):
#             X_train_split, X_val = X_train[train_index], X_train[test_index]
#             y_train_split, y_val = y_train[train_index], y_train[test_index]
#             # Train the model
#             model.fit(X_train_split, y_train_split)

#             # Make predictions on the validation set
#             y_pred = model.predict(X_val)
#             # Evaluate the model
#             r2 = round(r2_score(y_val, y_pred),3)
#             mse = round(mean_squared_error(y_val, y_pred),3)
#             mape = round(mean_absolute_percentage_error(y_val, y_pred),3)
            
#             results.append(mse)
            
#         # average_mse = np.mean(eval_results)
#         y_pred = model.predict(X_test)
#         y_pred = inverseTransform(y_pred.reshape(-1,1),scaler)

#         fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        
# runAll(626)

UFCodes_VER = {
    14 : 'RR', 16 : 'AP'
}

df_m, df_d, X, y = openAndSplit('RR')

# subsample é a qtde de instancias por arvore -> default 0.8
model = xgboost.XGBRFRegressor(n_samples=1000)


# scaler = MinMaxScaler()
# X, y = scaleData(X, y, scaler)

# X_train, X_test, y_train, y_test = X[:626], X[626:], y[:626], y[626:]

# tscv = TimeSeriesSplit(n_splits=5)
# eval_results = []
# for train_index, test_index in tscv.split(X_train):
#     X_train_split, X_val = X_train[train_index], X_train[test_index]
#     y_train_split, y_val = y_train[train_index], y_train[test_index]

#     # Train the model
#     model.fit(X_train_split, y_train_split)

#     # Make predictions on the validation set
#     y_pred = model.predict(X_val)

#     # Evaluate the model
#     mse = mean_squared_error(y_val, y_pred)
    
#     # Store evaluation result
#     eval_results.append(mse)

# average_mse = np.mean(eval_results)

# y_pred = model.predict(X_test)

# # print(r2_score(y_test, y_pred))
# # print(mean_squared_error(y_test, y_pred))
# # print(mean_absolute_percentage_error(y_test,y_pred))

# y_pred = inverseTransform(y_pred.reshape(-1,1),scaler)


# fullGraph(df_m, df_d,pred=True,y_pred=y_pred)

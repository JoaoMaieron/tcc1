
# https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 15 : 'PA',  17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}

#=================================================================================
# ABRINDO DADOS
#=================================================================================

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
# GRAFICO
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

def objective(trial, X_train, y_train, y_test):
    params = {
        'kernel' : trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        'C' : trial.suggest_loguniform('C', 1e-3, 1e3),
        'epsilon' : trial.suggest_float('epsilon', 1e-3, 1e1)
        }

    tscv = TimeSeriesSplit(n_splits=5)
    eval_results = []
    for train_index, test_index in tscv.split(X_train):
        X_train_split, X_val = X_train[train_index], X_train[test_index]
        y_train_split, y_val = y_train[train_index], y_train[test_index]
        
        svr_model = SVR(**params)
        svr_model.fit(X_train_split, y_train_split)
        
        y_pred = svr_model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        
        eval_results.append(mse)
        
    return np.mean(eval_results)

#=================================================================================
# EXECUCAO TODOS
#=================================================================================

import csv
def runAllWithTuning(train_size):
    
    study = optuna.create_study(direction='minimize')
    
    results = []
    scaler = MinMaxScaler()
    
    for uf in UFCodes.values():
        # Preparando dados
        df_m, df_d, X, y = openAndSplit(uf)
        X, y = scaleData(X, y, scaler)
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        
        # Roda tuning
        study.optimize(lambda trial: objective(trial, X_train, y_train, y_test), n_trials=100)
        # Obtem resultados
        best_params = study.best_params
        # print(f'Hiperparametros: {best_params}')        
        best_svr_model = SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])
        
        # Predicoes
        best_svr_model.fit(X_train, y_train)
        y_pred = best_svr_model.predict(X_test)
        
        # Metricas
        r2 = round(r2_score(y_test, y_pred),3)
        mse = round(mean_squared_error(y_test, y_pred),3)
        mape = round(mean_absolute_percentage_error(y_test, y_pred),3)
        
        # Printa grafico
        y_pred = inverseTransform(y_pred.reshape(len(y_pred),1), scaler)        
        fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        
        # fig = optuna.visualization.plot_optimization_history(study)
        # fig.show()
        
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parametros':best_params})
    
    # Salva as metricas num txt
    with open('metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAPE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

runAllWithTuning(train_size=626)


#=================================================================================
# UM SO
#=================================================================================

# df_m, df_d, X, y = openAndSplit('SP')

# scaler = MinMaxScaler()

# X, y = scaleData(X, y, scaler)


# train_size = 626
# X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# model = SVR(kernel='rbf')

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)


# y_pred = model.predict(X_test)
# y_pred = inverseTransform(y_pred.reshape(len(y_pred),1), scaler)

# fullGraph(df_m, df_d, pred=True,y_pred=y_pred,uf='SP')









# https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md

import utils
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 14 : 'RR', 15 : 'PA', 16 : 'AP',  17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}

# UFCodes = {
#     14 : 'RR', 16 : 'AP'
# }

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
def runAllWithTuning():
    # Instancia umas coisa
    study = optuna.create_study(direction='minimize')    
    results = []
    # Laco principal
    for uf in UFCodes.values():
        # Preparando dados
        df_m, df_d, X, y = utils.openAndSplit(uf)
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        train_size = len(df_m) - 52
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        y_index = df_m.index[train_size:]
        
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
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parametros':best_params})
        
        # Desfaz transformacoes pros graficos
        y_pred = utils.inverseTransform(y_pred.reshape(len(y_pred),1), scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        # Printa graficos
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
        
        # Salva o modelo        
        with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
            pickle.dump(best_svr_model, file)

    
    # Salva as metricas num txt
    with open('./TESTE/metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAPE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

# runAllWithTuning()

#=================================================================================
# CARREGAR O MODELO
#=================================================================================

def loadAllModels():
    results = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        train_size = len(df_m) - 52
        X_test, y_test = X[train_size:], y[train_size:]
        y_index = df_m.index[train_size:]
        
        # Carregando o modelo                    
        with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
            model = pickle.load(file)
                
        # Previsao
        y_pred = model.predict(X_test)        
        
        # Metricas
        r2 = round(r2_score(y_test, y_pred),3)
        mse = round(mean_squared_error(y_test, y_pred),3)
        mape = round(mean_absolute_percentage_error(y_test, y_pred),3)
        # params = model.get_params()
        
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape})
        # results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
    
    # Salva metricas em txt
    with open('./TESTE/metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAPE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

# loadAllModels()


# # Abre as coisa
# uf = 'RO'
# train_size = 626
# df_m, df_d, X, y = utils.openAndSplit('RO')
# results = []


# # Escala e divide em treino/teste
# scaler = MinMaxScaler()
# X, y = utils.scaleData(X, y, scaler)
# X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# # Roda tuning
# study = optuna.create_study(direction='minimize')
# study.optimize(lambda trial: objective(trial, X_train, y_train, y_test), n_trials=100)
# # Obtem resultados
# best_params = study.best_params
# best_svr_model = SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])
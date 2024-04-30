import utils
import csv
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold

UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 14 : 'RR', 15 : 'PA', 16 : 'AP',  17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}

#=================================================================================
# TUNING HIPERPARAMETROS
#=================================================================================

def objective(trial, X_train, y_train, y_test):
    params = {
        'kernel' : trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        'C' : trial.suggest_float('C', 1e-3, 1e3),
        'epsilon' : trial.suggest_float('epsilon', 1e-3, 1e1),
        }

    tscv = TimeSeriesSplit(n_splits=5)
    eval_results = []
    for train_index, test_index in tscv.split(X_train):
        X_train_split, X_val = X_train[train_index], X_train[test_index]
        y_train_split, y_val = y_train[train_index], y_train[test_index]
        
        svr_model = SVR(**params)
        svr_model.fit(X_train_split, y_train_split.ravel())
        
        y_pred = svr_model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)        
        eval_results.append(mse)
        
    return np.mean(eval_results)

#=================================================================================
# EXECUCAO DE TODOS OS PREDITORES
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
        study.optimize(lambda trial: objective(trial, X_train, y_train, y_test), n_trials=200)
        # Obtem resultados
        best_params = study.best_params
        # print(f'Hiperparametros: {best_params}')        
        best_svr_model = SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])
        
        # Predicoes
        best_svr_model.fit(X_train, y_train.ravel())
        y_pred = best_svr_model.predict(X_test)
        
        # Metricas
        r2, mse, mae = utils.calcMetricas(y_test, y_pred)
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAE': mae, 'parametros':best_params})
        
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
        fieldnames = ['estado','R2','MSE', 'MAE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    # Faz um boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/metricas.csv'))

#=================================================================================
# RODANDO TODOS COM TUNING DE HIPERPARAMETROS
#=================================================================================

def runOneWithTuning(uf,study):
    df_m, df_d, X, y = utils.openAndSplit(uf)
    scaler = MinMaxScaler()
    X, y = utils.scaleData(X, y, scaler)
    train_size = len(df_m) - 52
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    y_index = df_m.index[train_size:] 
    
    def objective(trial):
        params = {
            'kernel' : trial.suggest_categorical('kernel', ['linear','poly']),
            'C' : trial.suggest_float('C', 1e-3, 1e3),
            'epsilon' : trial.suggest_float('epsilon', 1e-3, 1e1),
            }
    
        tscv = TimeSeriesSplit(n_splits=5)
        eval_results = []
        for train_index, test_index in tscv.split(X_train):
            X_train_split, X_val = X_train[train_index], X_train[test_index]
            y_train_split, y_val = y_train[train_index], y_train[test_index]
            
            svr_model = SVR(**params)
            svr_model.fit(X_train_split, y_train_split.ravel())
            
            y_pred = svr_model.predict(X_val)
            
            mse = mean_squared_error(y_val, y_pred)        
            eval_results.append(mse)
            
        return np.mean(eval_results)

    
    study.optimize(objective, n_trials=200)    
    best_params = study.best_params
    final_model = SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])
    
    final_model.fit(X_train, y_train.ravel())    
    y_pred = final_model.predict(X_test)
    
    r2, mse, mae = utils.calcMetricas(y_test, y_pred)
    
    # Desfaz as transformacoes para printar os graficos depois
    y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
    y_test = utils.inverseTransform(y_test, scaler)
    # Printa grafico
    utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
    utils.partialGraph(y_test, y_pred, y_index, uf, save=True) 
    
    with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
        pickle.dump(final_model, file)
    
    return best_params, r2, mse, mae

def runAllWithTuning2():
    results = []
    study = optuna.create_study(direction='minimize')
    for uf in UFCodes.values():
        best_params, r2, mse, mae = runOneWithTuning(uf,study)
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAE': mae, 'parametros':best_params})
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    # Faz o boxplot
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))

#=================================================================================
# ENSEMBLE POR MACRORREGIAO
#=================================================================================

def ensembleSimples(t=False):
    results = []
    for uf in UFCodes.values():
        print('Iniciando ',uf,'...')
        # Prepara o dataset do estado
        df_m, df_d, X, y = utils.openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        
        if t:
            selector = VarianceThreshold(0.01)
            X = selector.fit_transform(X)
        
        train_size = len(df_m) - 52
        X_test, y_test = X[train_size:], y[train_size:]
        y_index = df_m.index[train_size:]
        # Cria lista pra guardar as predicoes que vao ser feitas
        predictions = []
        for regiao in utils.regioes:
            # Procura o estado nas regioes
            if uf in regiao:                
                for estado in regiao:                    
                    # Faz predicao com os modelos de todos estados daquela regiao
                    with open('./backup_modelos/'+estado+'.pkl', 'rb') as file:
                        model = pickle.load(file)
                    print('Carregado modelo de ',estado)
                    predictions.append(model.predict(X_test))
                # Faz a media das previsoes
                ensemble_predictions = np.array(predictions)
                ensemble_predictions = np.mean(ensemble_predictions,axis=0)
        # No final, metricas das previsoes feitas
        r2, mse, mae = utils.calcMetricas(y_test, ensemble_predictions)
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAE' : mae})
        # Graficos das previsoes
        y_pred = utils.inverseTransform(ensemble_predictions.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
    # Salva metricas em txt
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r) 
    # Faz o boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'),ensemble=True)

#=================================================================================
# TESTE DE OVERFITTING
#=================================================================================

def overfitTest(t=False):
    results = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        
        if t:
            selector = VarianceThreshold()
            X = selector.fit_transform(X)
        
        train_size = len(df_m) - 52
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        y_index = df_m.index[:train_size]
        
        with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
            model = pickle.load(file)
        
        y_pred = model.predict(X_train)        
        
        # Metricas
        r2, mse, mae = utils.calcMetricas(y_train, y_pred)
        # params = model.get_params()
        
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAE': mae})
        # results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_train, scaler)
        
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf,gambiarra=True)
        
    # Salva metricas em txt
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    # Faz um boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))

#=================================================================================

def loadAllModels(train = False):
    results = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)        
        
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        train_size = len(df_m) - 52
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        y_index = df_m.index[train_size:]
               
        # Carregando o modelo                    
        with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
            model = pickle.load(file)
        
        if train:
            model.fit(X_train, y_train)
        
        # Previsao
        y_pred = model.predict(X_test)        
        
        # Metricas
        r2, mse, mae = utils.calcMetricas(y_test, y_pred)
        # params = model.get_params()
        
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAE': mae})
        # results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
    
    # Salva metricas em txt
    with open('./TESTE/metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    # Faz um boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/metricas.csv'))
    
#=================================================================================
# CHAMADAS DE FUNCAO
#=================================================================================

# runAllWithTuning()
# loadAllModels(train=False)
# ensembleSimples()
# overfitTest()
# multiTrain()

import utils
import xgboost
import optuna
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
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

# print(xgboost.__version__)

# https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

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
def runAllWithTuning():
    # Instancia umas coisas
    study = optuna.create_study(direction='minimize')    
    results = []
    # Laco principal
    for uf in UFCodes.values():
        # Preparo dos dados
        df_m, df_d, X, y = utils.openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        train_size = len(df_m) - 52 # Isso aqui é só pq RR nao tem o mesmo tamanho que os demais
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        y_index = df_m.index[train_size:]
        
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
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parametros':best_params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
        
        # Salva o modelo
        final_model.save_model('./backup_modelos/'+uf+'.model')
    
    # Salva metricas em txt
    with open('./TESTE/metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAPE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

# runAllWithTuning()

#=================================================================================
# RODANDO TODOS SEM TREINO
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
        model = xgboost.XGBRFRegressor()
        model.load_model('./backup_modelos/'+uf+'.model')
        
        # Previsao
        y_pred = model.predict(X_test)        
        
        # Metricas
        r2 = round(r2_score(y_test, y_pred),3)
        mse = round(mean_squared_error(y_test, y_pred),3)
        mape = round(mean_absolute_percentage_error(y_test, y_pred),3)
        params = model.get_params()
        # params = model.get_xgb_params()
        # Registra metricas
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
    
    # Salva metricas em txt
    with open('./TESTE/metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

# loadAllModels()


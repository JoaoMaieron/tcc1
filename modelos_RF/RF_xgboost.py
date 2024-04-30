import utils
import csv
import xgboost
import optuna
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

UFCodes = {
    11 : 'RO', 12 : 'AC', 13 : 'AM', 14 : 'RR', 15 : 'PA', 16 : 'AP',  17 : 'TO', 
    21 : 'MA', 22 : 'PI', 23 : 'CE', 24 : 'RN', 25 : 'PB', 26 : 'PE', 27 : 'AL', 28 : 'SE', 29 : 'BA',
    31 : 'MG', 32 : 'ES', 33 : 'RJ', 35 : 'SP', 
    41 : 'PR', 42 : 'SC', 43 : 'RS',
    50 : 'MS', 51 : 'MT', 52 : 'GO', 53 : 'DF'
}

# print(xgboost.__version__)
# https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

#=================================================================================
# TUNING DE HIPERPARAMETROS
#=================================================================================

# https://forecastegy.com/posts/multiple-time-series-forecasting-with-scikit-learn/
# https://forecastegy.com/posts/multiple-time-series-forecasting-with-xgboost-in-python/#how-to-prepare-time-series-data-for-xgboost-in-python
def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500,step=50),
        'max_depth' : trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.50, 0.9),
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

        mae = mean_absolute_error(y_val, y_pred)
        eval_results.append(mae)
        
    return np.mean(eval_results) # Pode-se alterar a metrica de referencia aqui

# import csv
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
        y_index = df_m.index[train_size:]
        
        # Para diminuir o training set
        # X_train, X_test, y_train, y_test = X[:train_size-152], X[train_size:], y[:train_size-152], y[train_size:]
        # Training set original de 626 semanas
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
        r2, mape, mse, mae = utils.calcMetricas(y_test, y_pred)
        # Registra metricas
        results.append({'estado': uf,'R2':r2, 'MSE': mse, 'MAE': mae,'MAPE':mape, 'parametros':best_params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)       
         
        # Salva o preditor
        with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
            pickle.dump(final_model, file)
        # final_model.save_model('./backup_modelos/'+uf+'.model')
    
        # Salva metricas em txt
    with open('./TESTE/metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2', 'MSE', 'MAE','MAPE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    # Faz o boxplot
    utils.boxplot(pd.read_csv('./TESTE/metricas.csv'))
    
#=================================================================================
# CONSTRUIR MODELO TUNANDO
#=================================================================================

# def runOneWithTuning(uf):
#     df_m, df_d, X, y = utils.openAndSplit(uf)        
#     scaler = MinMaxScaler()
#     X, y = utils.scaleData(X, y, scaler)
#     train_size = len(df_m) - 52
#     y_index = df_m.index[train_size:]
#     X_train, X_test, y_train, y_test = X[:542], X[train_size:], y[:542], y[train_size:]
    
#     def objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300,step=50),
#             'max_depth' : trial.suggest_int('max_depth', 3, 7),
#             'subsample': trial.suggest_float('subsample', 0.5, 0.8),
#             'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 0.9),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
#             }
#         tscv = TimeSeriesSplit(n_splits=5)
#         eval_results = []
#         for train_index, test_index in tscv.split(X_train):
#             X_train_split, X_val = X_train[train_index], X_train[test_index]
#             y_train_split, y_val = y_train[train_index], y_train[test_index]
            
#             model = xgboost.XGBRFRegressor(**params)        
#             model.fit(X_train_split, y_train_split)
            
#             y_pred = model.predict(X_val)

#             mae = mean_absolute_error(y_val, y_pred)
#             eval_results.append(mae)
            
#             # r2 = r2_score(y_val, y_pred)
#             # eval_results.append(r2)
            
#         return np.mean(eval_results)

#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=50)    
#     best_params = study.best_params
#     final_model = xgboost.XGBRFRegressor(**best_params,xgb_model=None)
    
#     final_model.fit(X_train, y_train)    
#     y_pred = final_model.predict(X_test)
    
#     r2, mse, mae = utils.calcMetricas(y_test, y_pred)
    
#     # Desfaz as transformacoes para printar os graficos depois
#     y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
#     y_test = utils.inverseTransform(y_test, scaler)
#     # Printa grafico
#     utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
#     utils.partialGraph(y_test, y_pred, y_index, uf, save=True) 
    
#     with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
#         pickle.dump(final_model, file)
    
#     return best_params, r2, mse, mae

# def runAllWithTuning2():
#     results = []
#     for uf in UFCodes.values():
#         best_params, r2, mse, mae = runOneWithTuning(uf)
#         results.append({'estado': uf, 'MSE': mse, 'MAE': mae,'MAPE':mape, 'parametros':best_params})
#     with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
#         fieldnames = ['estado','MSE', 'MAE','MAPE', 'parametros']
#         writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
#         writer.writeheader()
#         for r in results:
#             writer.writerow(r)
#     # Faz o boxplot
#     utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))

#=================================================================================
# CARREGANDO E RODANDO TODOS OS PREDITORES
#=================================================================================

def loadAllModels(train=False):
    results = []
    preds = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        
        train_size = len(df_m) - 52
        
        # Para diminuir o training set
        # X_train, X_test, y_train, y_test = X[:train_size-52], X[train_size:], y[:train_size-52], y[train_size:]
        # Training set original de 626 semanas
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        
        y_index = df_m.index[train_size:]
        
        # Carregando o modelo
        with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
            model = pickle.load(file)
        # model = xgboost.XGBRFRegressor()
        # model.load_model('./backup_modelos/'+uf+'.model')
        
        if train:
            model.fit(X_train, y_train)
            
        # Previsao
        y_pred = model.predict(X_test)        

        # Metricas
        # params = model.get_params()
        # params = model.get_xgb_params()
        # Registra metricas
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        
        preds.append(y_pred)
        
        r2, mape, mse, mae = utils.calcMetricas(y_test, y_pred)
        r2 = round(r2_score(y_test, y_pred),3)
        mape =round(mean_absolute_percentage_error(y_test, y_pred),3)
        
        results.append({'estado': uf, 'R2':r2, 'MSE': mse, 'MAE': mae, 'MAPE': mape})
        # results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
        
        if train:
            with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
                pickle.dump(model, file)
        
        print('ACABOSU ',uf)
    
    # Salva metricas em txt
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        fieldnames = ['estado','R2','MSE', 'MAE','MAPE']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))
    
    return preds

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
        
        # Graficos das previsoes
        y_pred = utils.inverseTransform(ensemble_predictions.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        
        r2, mape, mse, mae = utils.calcMetricas(y_test, y_pred)
        r2 = round(r2_score(y_test, y_pred),3)
        mape =round(mean_absolute_percentage_error(y_test, y_pred),3)
        results.append({'estado': uf, 'R2':r2, 'MSE': mse, 'MAE': mae, 'MAPE': mape})
        
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
    # Salva metricas em txt
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:

        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        fieldnames = ['estado','R2','MSE', 'MAE','MAPE']

        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r) 
    # Faz o boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'),ensemble=True)


#=================================================================================
# RODAR OS PREDITORES SOBRE CONJUNTOS DE TESTE PARA VER SE HOUVE OVERFITTING
#=================================================================================

def overfitTest(t=False):
    results = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        
        if t:
            selector = VarianceThreshold(0.01)
            X = selector.fit_transform(X)
        
        train_size = len(df_m) - 52
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        y_index = df_m.index[:train_size]
        
        with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
            model = pickle.load(file)
        
        y_pred = model.predict(X_train)        
        
        # Metricas
        mape, mse, mae = utils.calcMetricas(y_train, y_pred)
        # params = model.get_params()
        
        # Registra metricas
        results.append({'estado': uf, 'MSE': mse, 'MAE': mae, 'MAPE': mape})
        # results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_train, scaler)
        
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf,gambiarra=True)
        
    # Salva metricas em txt
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','MSE', 'MAE','MAPE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    # Faz um boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))

#=================================================================================
# CORRECAO DE OVERFITTING: 
#=================================================================================

# def multiTrain():
#     results=[]
#     for uf in UFCodes.values():
#         # Abre e prepara o dataset da uf em si
#         df_m, df_d, X, y = utils.openAndSplit(uf)
#         scaler = MinMaxScaler()
#         X, y = utils.scaleData(X, y, scaler)
#         train_size = len(df_m) - 52
#         X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
#         y_index = df_m.index[train_size:]        
#         # Abre o modelo da uf
#         with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
#             model = pickle.load(file)
#         for r in utils.regioes:
#             if uf in r:
#                 for uf2 in r:
#                     if uf2 != uf:
#                         # Abre pra todas as outras uf da mesma regiao e treina
#                         df_m2, df_d2, X2, y2 = utils.openAndSplit(uf2)
#                         X2, y2 = utils.scaleData(X2, y2, scaler)
#                         X_train2, y_train2, = X2[:train_size], y2[:train_size]
#                         model.fit(X_train2,y_train2)
        
#         model.fit(X_train,y_train)
#         y_pred = model.predict(X_test)
        
#         # Metricas
#         mape, mse, mae = utils.calcMetricas(y_test, y_pred)
#         # Registra metricas
#         results.append({'estado': uf, 'MSE': mse, 'MAE': mae, 'MAPE': mape})
        
#         # Desfaz as transformacoes para printar os graficos depois
#         y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
#         y_test = utils.inverseTransform(y_test, scaler)
#         # Printa grafico
#         utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
#         utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
     
#     with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
#         fieldnames = ['estado','MSE', 'MAE','MAPE', 'parametros']
#         writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
#         writer.writeheader()
#         for r in results:
#             writer.writerow(r)
#     # Faz o boxplot
#     utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))


#==============================================================================
# TESTES SOBRE AS METRICAS, DESCONSIDERAR
#==============================================================================

# df = pd.read_csv('_metricas_LSTM.csv')
# df['MSE Rank'] = df['MSE'].rank(ascending=True)
# df['MAE Rank'] = df['MAE'].rank(ascending=True)
# df['MAPE Rank'] = df['MAPE'].rank(ascending=True)
# df['R2 Rank'] = df['R2'].rank(ascending=False)
# df['Average Rank'] = df[['MAPE Rank', 'R2 Rank']].mean(axis=1)

# df.to_csv('_rankingsRF.csv',index=False)

# med = np.mean(df['MAE'])
# print(med)
# LSTM
# mse 0.0378
# mae 0.1097
# r2 -4.4327

# RF
# mse 0.0395
# mae 0.112
# r2 -0.6201

# SVR
# mse 0.0476
# mae 0.1249
# r2 -0.5652

# MAE mais baixo - am, tj, es, ap, mg, pa, ba
# MSE mais baixo - am, rj, es, ap, mg, ms, pa, 
# R2 positivo    - ba, pr, ms, sp, pa, mg, al, pe, ap, rn

#==============================================================================
# BENCHMARK MODELO NAIVE
#==============================================================================

def naive():
    results=[]
    for uf in UFCodes.values():
        # Abre e prepara o dataset da uf em si
        df_m, df_d, X, y = utils.openAndSplit(uf)
        # scaler = MinMaxScaler()
        # X, y = utils.scaleData(X, y, scaler)
        train_size = len(df_m) - 52
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        y_index = df_m.index[train_size:]
        
        # df_d['a'] = df_d['casos'].shift(1)        
        # y_pred =df_d['a'][train_size:]
        
        df_d['a'] = df_d['casos'].rolling(window=26).mean()
        y_pred =df_d['a'][train_size:]
        
        # Metricas
        r2, mape, mse, mae = utils.calcMetricas(y_test, y_pred)
        # Registra metricas
        results.append({'estado': uf, 'R2':r2, 'MSE': mse, 'MAE': mae, 'MAPE': mape})
        
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
     
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE','MAE','MAPE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    # Faz o boxplot
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))

#=================================================================================
# CHAMADAS DE FUNCAO
#=================================================================================

# runAllWithTuning()      
# runAllWithTuning2()
# preds_rf = loadAllModels(train=False)
# ensembleSimples()
# overfitTest()
# naive()
# multiTrain()

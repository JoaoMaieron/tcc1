import matplotlib.dates as mdates
import utils
import csv
import optuna
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

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

def objective(trial, X_train, X_test, y_train, y_test):       
    tscv = TimeSeriesSplit(n_splits=5)
    eval_results = []
    
    for train_index, test_index in tscv.split(X_train):
        X_train_split, X_val = X_train[train_index], X_train[test_index]
        y_train_split, y_val = y_train[train_index], y_train[test_index]
        
        # model = create_lstm_model(trial, X_train_split)
        
        # Montagem do modelo
        model = Sequential()        

        # DESCOMENTAR ABAIXO PARA OTIMIZAR MAIS COISAS
        model.add(LSTM(units=trial.suggest_int('units', 64, 256, step=32),input_shape=(X_train_split.shape[1], X_train_split.shape[2]),return_sequences=True))
        model.add(Dropout(rate=0.2))
        # model.add(LSTM(units=trial.suggest_int('units_layer_0', 64, 256,step=32),activation='relu',return_sequences=True))
        # model.add(Dropout(rate=0.2))
        # model.add(LSTM(units=trial.suggest_int('units_layer_1', 64, 256,step=32),activation='relu',return_sequences=True))
        # model.add(Dropout(rate=0.2))
        # model.add(LSTM(units=trial.suggest_int('units_layer_2', 64, 256,step=32),activation='relu',return_sequences=True))
        # model.add(Dropout(rate=0.2))
        model.add(LSTM(units=trial.suggest_int('units_layer_3', 64, 256,step=32)))
        
        # OTIMIZANDO NUMERO DE CAMADAS OCULTAS
        # for i in range(trial.suggest_int('num_layers', 1, 3)):
        #     return_sequences = i < trial.suggest_int('num_layers', 1, 3) - 1
        #     model.add(LSTM(
        #         units=trial.suggest_int(f'units_layer_{i}', 64, 256,step=32),
        #         return_sequences=return_sequences
        #     ))
        
        # DESCOMENTAR ABAIXO PARA MANTER FIXOS OS HIPERPARAMETROS
        # model.add(LSTM(units=192,input_shape=(X_train_split.shape[1], X_train_split.shape[2]),return_sequences=True))        
        # model.add(Dropout(rate=0.2))
        # model.add(LSTM(units=224,return_sequences=True))
        # model.add(Dropout(rate=0.2))
        # model.add(LSTM(units=96,return_sequences=True))
        # model.add(Dropout(rate=0.2))
        # model.add(LSTM(units=128))
        # model.add(Dense(units=1))
        
        # model.add(LSTM(units=128))
        model.add(Dense(units=1))
        
        # Compila e roda o modelo
        model.compile(
            optimizer=Adam(learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3)),
            loss='mean_squared_error'            
        )    
        batch_size = trial.suggest_int('batch_size', 64, 128, step=32)        
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)        
        model.fit(X_train_split, y_train_split, epochs=10, batch_size=batch_size,callbacks=[early_stopping],verbose=0)

        y_pred = model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        
        eval_results.append(mse)
        
    return np.mean(eval_results)

#=================================================================================
# RODANDO TODOS OS PREDITORES COM TUNING
#=================================================================================

def runAllWithTuning(WS,t=False):
    '''
    Roda tuning de hiperparametros para todos os datasets, salvando as metricas dos resultados
    
    Args:
        WS : int, tamanho do lookback
    '''
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    
    results = []
    scaler = MinMaxScaler()
    for uf in UFCodes.values():
        # Preparando dados
        df_m, df_d, X, y = utils.openAndSplit(uf)
        X, y = utils.scaleData(X, y, scaler)
        
        # Eliminando atributos com pouca variacao        
        # if t:
        #     selector = VarianceThreshold(0.01)
        #     X = selector.fit_transform(X)
        
        # train_size = len(df_m) - 65 # para ws=13
        train_size = len(df_m) - 78 # para ws=26
        X_train, X_test, y_train, y_test = utils.prepareData(WS, train_size, X, y)        
        y_index = df_m.index[len(df_m)-52:]
        
        # tuning
        study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=15)
        best_params = study.best_params
        
        final_model = Sequential()
        # final_model.add(LSTM(units=192,input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))        
        # final_model.add(Dropout(rate=0.2))
        # final_model.add(LSTM(units=224,return_sequences=True))
        # final_model.add(Dropout(rate=0.2))
        # final_model.add(LSTM(units=96,return_sequences=True))
        # final_model.add(Dropout(rate=0.2))
        # final_model.add(LSTM(units=128,return_sequences=True))
        # final_model.add(Dropout(rate=0.2))
        # final_model.add(LSTM(units=128))
        # final_model.add(Dense(units=1))
        
        
        final_model.add(LSTM(units=best_params['units'],input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))      
        final_model.add(Dropout(rate=0.2))
        # final_model.add(LSTM(units=best_params['units_layer_0'],activation='relu',return_sequences=True))
        # final_model.add(Dropout(rate=0.2))
        # final_model.add(LSTM(units=best_params['units_layer_1'],activation='relu',return_sequences=True))
        # final_model.add(Dropout(rate=0.2))
        # final_model.add(LSTM(units=best_params['units_layer_2'],activation='relu',return_sequences=True))
        # final_model.add(Dropout(rate=0.2))
        final_model.add(LSTM(units=best_params['units_layer_3']))
        final_model.add(Dense(units=1))
        
        # CASO TENHA OTIMIZADO NUM DE CAMADAS OCULTAS
        # for i in range(best_params['num_layers']):
        #     final_model.add(LSTM(
        #         units=best_params[f'units_layer_{i}'],
        #         activation='relu',
        #         return_sequences=(i < best_params['num_layers'] - 1)  # return sequence nao pode na ultima
        #     ))
        # final_model.add(Dense(units=1))

        # Treina o  modelo final        
        final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)        
        final_model.fit(X_train,y_train,epochs=300,batch_size=best_params['batch_size'],callbacks=[early_stopping],verbose=2)
        
        # Predicoes
        y_pred = final_model.predict(X_test)
        
        # Metricas - com failsafe para os que estava saindo nan na previsao
        if not np.isnan(y_pred).any():
            
            r2, mse, mae, mape = utils.calcMetricas(y_test, y_pred)            
        
            y_pred = utils.inverseTransform(y_pred.reshape(len(y_pred),1), scaler)
            y_test = utils.inverseTransform(y_test, scaler)
            utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
            utils.partialGraph(y_test, y_pred.reshape(-1), y_index, uf, save=True)
        else:
            r2, mse, mae = 0, 0, 0
        
        results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAE': mae,'MAPE':mape, 'parametros' : best_params})
        
        with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
            pickle.dump(final_model, file)
        
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE','MAPE', 'parametros']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        
    # Faz o boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))
    
#=================================================================================
# CARREGAR O MODELO
#=================================================================================
def loadAllModels(WS,train=False,novos=False):
    '''
    Carrega todos os modelos para repetir as predicoes e gerar metricas. Pode treinar eles denovo
    
    Args:
        WS : int, lookback window
        train : bool, se 1 monta e treina novos modelos, se 0 so carrega modelos pre-treinados
    '''
    results = []
    preds = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        
        a = 52 + WS
        # train_size = len(df_m)        
        # train_size = len(df_m) - a # Para ws=13
        train_size = len(df_m) - 78 # Para ws=26
        # train_size = len(df_m) - 104 # Para ws=52
        X_train, X_test, y_train, y_test = utils.prepareData(WS, train_size, X, y) 
        y_index = df_m.index[len(df_m)-52:]
        
        if train and novos:
           
            # Treinando um modelo novo
            model = Sequential()
            model.add(LSTM(units=192,activation='tanh',input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))        
            model.add(Dropout(rate=0.8))
            model.add(LSTM(units=224,activation='tanh',return_sequences=True))
            model.add(Dropout(rate=0.5))
            model.add(LSTM(units=224,activation='tanh',return_sequences=True))
            model.add(Dropout(rate=0.5))
            # model.add(LSTM(units=224,activation='tanh',return_sequences=True))
            # model.add(Dropout(rate=0.5))
            # model.add(LSTM(units=224,activation='tanh',return_sequences=True))
            # model.add(Dropout(rate=0.5))
            model.add(LSTM(units=128))
            model.add(Dense(units=1))
            
            model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error') 
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train,y_train,epochs=50,batch_size=96,callbacks=[early_stopping],verbose=2)
            
            # plt.ioff()
            # plt.figure()
            # plt.plot(history.history['loss'], label='Training Loss')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.legend()
            # # plt.show()
            # plt.savefig(fname = './TESTE/loss/'+uf+'.png',format='png')
            
        else:
            # Carregando um modelo pre-existente
            with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
                model = pickle.load(file)
            
            if train and not novos:
                early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                model.fit(X_train,y_train,epochs=50,batch_size=64,callbacks=[early_stopping],verbose=2)
                with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
                    pickle.dump(model, file)
            
            # model.summary() # Carrega resumo do modelo
        
        # Previsao
        y_pred = model.predict(X_test)        
        
        # Metricas
        # params = model.get_params()
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        
        preds.append(y_pred)

        r2, mape, mse, mae = utils.calcMetricas(y_test, y_pred)
        
        mape = round(mean_absolute_percentage_error(y_test, y_pred),3 ) 

        # Registra metricas
        results.append({'estado': uf, 'R2':r2,'MSE': mse, 'MAE': mae, 'MAPE': mape})
        # results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # y_pred
        # y_test
        # mean_absolute_error(y_test, y_pred)
        # mean_squared_error(y_test, y_pred)
        # mean_absolute_percentage_error(y_test, y_pred)
        # r2_score(y_test, y_pred)
        
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf,WS=WS)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
        
        if train:
            with open('./backup_modelos/'+uf+'.pkl', 'wb') as file:
                pickle.dump(model, file)

    # Salva metricas em txt - um pra cada estado pq tenho medo de dar problema em algum 
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE','MAPE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)        
    
    # Faz o boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))
    
    return preds

#=================================================================================
# ENSEMBLE POR MACRORREGIAO
#=================================================================================

def ensembleSimples():
    '''
    Carrega todos os preditores e executa ensemble simples
    '''
    results = []
    for uf in UFCodes.values():
        print('Iniciando ',uf,'...')
        # Prepara o dataset do estado
        df_m, df_d, X, y = utils.openAndSplit(uf)        
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        
        # train_size = len(df_m) - 65 # para ws=13
        train_size = len(df_m) - 78 # para ws=26
        X_train, X_test, y_train, y_test = utils.prepareData(26, train_size, X, y) # alterar a WS aqui
        y_index = df_m.index[len(df_m)-52:]
        
        # X_test2 = X[len(df_m)-52:]
        
        # Cria lista pra guardar as predicoes que vao ser feitas
        predictions = []
        for regiao in utils.regioes:
            # Procura o estado nas regioes
            if uf in regiao:                
                for estado in regiao:                    
                    # Faz predicao com os modelos de todos estados daquela regiao
                    with open('./modelos_lstm/'+estado+'.pkl', 'rb') as file:
                        model = pickle.load(file)
                    # print('Carregado modelo de ',estado)
                    predictions.append(model.predict(X_test))
                    # a = model.predict(X_test)
                    # a = a.flatten()
                    # predictions.append(a)
                    
                    # with open('./modelos_svr/'+estado+'.pkl', 'rb') as file:
                    #     model = pickle.load(file)
                    # predictions.append(model.predict(X_test2))
                    
                    # with open('./modelos_rf/'+estado+'.pkl', 'rb') as file:
                    #     model = pickle.load(file)
                    # predictions.append(model.predict(X_test2))
                    
                # Faz a media das previsoes
                ensemble_predictions = np.array(predictions)
                ensemble_predictions = np.mean(ensemble_predictions,axis=0)
                # ensemble_predictions = np.mean(predictions,axis=0)
        # No final, metricas das previsoes feitas
        # r2, mape, mse, mae = utils.calcMetricas(y_test, ensemble_predictions)
        
        # Graficos das previsoes
        y_pred = utils.inverseTransform(ensemble_predictions.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_test, scaler)
        
        r2, mape, mse, mae = utils.calcMetricas(y_test, y_pred)
        mape = round(mean_absolute_percentage_error(y_test, y_pred),3 ) 
        results.append({'estado': uf,'R2':r2, 'MSE': mse, 'MAE': mae,'MAPE':mape})
        
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf)
        utils.partialGraph(y_test, y_pred, y_index, uf, save=True)
    # Salva metricas em txt
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE','MAPE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r) 
    # Faz o boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'),ensemble=True)

#================================================================================
# TESTA SE NAO DEU OVERFIT
#================================================================================

def overfitTest(t=False,WS=26):
    '''
    Carrega os preditores e executa-os sobre conjuntos de treino, para verificar a existencia de overfitting
    Args:
        WS : int, tamanho da janela de lookback
    '''
    results = []
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)
        scaler = MinMaxScaler()
        X, y = utils.scaleData(X, y, scaler)
        
        # if t:
        #     selector = VarianceThreshold()
        #     X = selector.fit_transform(X)
        
        # train_size = len(df_m) - 65 # para ws=13
        train_size = len(df_m) - 78 # para ws=26
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        X_train, X_test, y_train, y_test = utils.prepareData(WS, train_size, X, y)  #
        y_index = df_m.index[:train_size]
        
        with open('./backup_modelos/'+uf+'.pkl', 'rb') as file:
            model = pickle.load(file)
        
        y_pred = model.predict(X_train)        
        
        # Metricas
        r2, mape, mse, mae = utils.calcMetricas(y_train, y_pred)
        # params = model.get_params()
        
        # Registra metricas
        results.append({'estado': uf, 'R2':r2, 'MSE': mse, 'MAE': mae,'MAPE':mape})
        # results.append({'estado': uf, 'R2': r2, 'MSE': mse, 'MAPE' : mape, 'parameters' : params})
        
        # Desfaz as transformacoes para printar os graficos depois
        y_pred = utils.inverseTransform(y_pred.reshape(-1,1),scaler)
        y_test = utils.inverseTransform(y_train, scaler)
        
        # Printa grafico
        utils.fullGraph(df_m, df_d,pred=True,y_pred=y_pred,save=True,uf=uf,gambiarra=True)
        
    # Salva metricas em txt
    with open('./TESTE/_metricas.csv', 'w',newline='') as csvfile:
        fieldnames = ['estado','R2','MSE', 'MAE','MAPE']
        # fieldnames = ['estado','R2','MSE', 'MAPE', 'parameters']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    # Faz um boxplot das metricas
    utils.boxplot(pd.read_csv('./TESTE/_metricas.csv'))

#================================================================================
# SEASONAL DECOMP
#================================================================================

# from statsmodels.tsa.seasonal import seasonal_decompose
# import matplotlib.pyplot as plt

# def seasonalDecomp():
#     df = pd.read_csv('../dados_versao_final/meteorologicos/RS.csv',index_col='data')
#     # Apply seasonal_decompose
#     result = seasonal_decompose(df['vento'], model='additive', period=52)
    
#     # Plot the decomposed components
#     plt.figure(figsize=(12, 8))
#     result.plot()
#     plt.show()

#=================================================================================
# PLOTAR GRAFICO DE UMA FEATURE SÓ
#=================================================================================

def a():
    plt.ioff()
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax.plot(df_m.index,df_d['casos'], label='Casos')
        ax.plot(df_m.index,df_m['temp_max'], label='Máximas em ºC')
        ax.plot(df_m.index,df_m['temp_media'], label='Médias em ºC')
        ax.plot(df_m.index,df_m['temp_min'], label='Mínimas em ºC')
        # ax.set_ylabel('Casos')
        ax.set_title(uf+' - temperaturas')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fname = './g/'+uf+'_temps.png',format='png')
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax.plot(df_m.index,df_d['casos'], label='Casos')
        ax.plot(df_m.index,df_m['umid_media'], label='Médias por semana')
        ax.plot(df_m.index,df_m['umid_min'], label='Mínimas por semana')
        # bb = pd.read_csv('MAE_MSE.csv',index_col='e')
        # ax.set_ylabel('Casos')
        ax.set_title(uf+' - umidade relativa do ar')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fname = './g/'+uf+'_umid.png',format='png')
        # plt.show()
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.plot(df_m.index,df_m['prec'], label='Acúmulo por semana')
        ax.set_title(uf+' - precipitação em mm')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fname = './g/'+uf+'_prec.png',format='png')
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.plot(df_m.index,df_m['vento_max'], label='Máximas')
        ax.plot(df_m.index,df_m['vento_media'], label='Médias')
        ax.set_title(uf+' - rajadas de vento, em m/s')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fname = './g/'+uf+'_vento.png',format='png')

#=================================================================================
# GRAFICOS BPNITÒ
#=================================================================================

def graficoBonito():
    plt.ioff()
    for uf in UFCodes.values():
        df_m, df_d, X, y = utils.openAndSplit(uf)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax.plot(df_m.index,df_d['casos'], label='Casos')
        a = df_d['casos'][len(df_d)-52:]
        b = df_d['casos'][:len(df_d)-52]
        ax.plot(df_m.index[len(df_d)-52:], a, color='blue', label='Período de teste')
        ax.plot(df_m.index[:len(df_d)-52], b, color='green')
        ax.grid(True)
        ax.set_title('Casos por semana - '+uf)
        ax.set_ylabel('Casos')
        ax.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(fname = uf+'_bonito.png',format='png')
        plt.close(fig)
        
#=================================================================================
# # PLOTAR A MATRIZ DE CORRELACAO
#=================================================================================

# import seaborn as sns
# df_m, df_d, X, y = utils.openAndSplit('RO')
# matriz = df_m.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(matriz, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Correlation Matrix with Highlighted Features')
# plt.show()

#=================================================================================
# TESTAR UM PRA DEBUGAR
#=================================================================================

# df_m, df_d, X, y = utils.openAndSplit('RR')        
# scaler = MinMaxScaler()
# X, y = utils.scaleData(X, y, scaler)
# train_size = len(df_m) - 78 # Para ws=26
# # train_size = len(df_m) - 104 # Para ws=52
# X_train, X_test, y_train, y_test = prepareData(26, train_size, X, y) 
# y_index = df_m.index[len(df_m)-52:]


# model = Sequential()        
# model.add(LSTM(units=192,input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))        
# model.add(Dropout(rate=0.2))
# model.add(LSTM(units=192,return_sequences=True))
# model.add(Dropout(rate=0.2))
# model.add(LSTM(units=224,return_sequences=True))
# model.add(Dropout(rate=0.2))
# model.add(LSTM(units=224))
# model.add(Dense(units=52))
# model.compile(optimizer=Adam(learning_rate=0.00001),loss='mean_squared_error') 

# with open('./backup_modelos/RR.pkl', 'rb') as file:
#     model = pickle.load(file)

# early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
# model.fit(X_train,y_train,epochs=10,batch_size=32,callbacks=[early_stopping],verbose=2)

# # Previsao
# y_pred = model.predict(X_test)       

# # Metricas
# r2, mse, mae, mape = utils.calcMetricas(y_test, y_pred)

# model.summary()

#================================================================================
# GRAFICOS ADICIONAIS
#================================================================================

# count = 0
# for uf in UFCodes.values():
#     df_m, df_d, X, y = utils.openAndSplit(uf)
#     train_size = len(df_m)-52
#     # X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
#     y_index = df_m.index[train_size:]
    
    
#     plt.ioff()
#     fig, ax = plt.subplots(figsize=(10,6))
#     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     ax.plot(y_index,y[train_size:], label='Reais')    
#     ax.plot(y_index,preds_lstm[count], label='LSTM - previstos')
#     ax.plot(y_index,preds_svr[count], label='SVR - previstos')
#     ax.plot(y_index,preds_rf[count], label='RF - previstos')
    
#     ax.grid(True)
#     ax.set_title('Previsões - '+uf,fontsize=13)
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(fname = uf+'_preds.png',format='png')
#     plt.close()
    
#     count = count + 1

# tudo = pd.read_csv('ens_doido.csv')
# rf = pd.read_csv('rf.csv')
# svr = pd.read_csv('svr.csv')
# lstm = pd.read_csv('lstm.csv')
# ls_sv = pd.read_csv('as.csv')

#================================================================================
# CHAMADAS DE FUNCAO
#================================================================================

# runAllWithTuning(WS=26)

# loadAllModels(WS=26,train=True,novos=True)
# preds_lstm = loadAllModels(WS=26,train=False,novos=False)

# ensembleSimples()

# overfitTest()

# graficoBonito()
# a = pd.read_csv('_metricas.csv')




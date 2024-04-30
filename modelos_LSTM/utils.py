import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import VarianceThreshold

#==============================================================================
# VARS GLOBAIS
#==============================================================================

regioes = [
    ['RO','AC','AM','RR','PA','AP','TO' ],
    ['MA','PI','CE','RN','PB','PE','AL','SE','BA'],
    ['MG','ES','RJ','SP'],
    ['PR','SC','RS'],
    ['MS','MT','GO','DF']
    ]

#==============================================================================
# PRE PROCESS
#==============================================================================

def openAndSplit(uf):
    '''
    Abre e prepara dados pra alimentar pros modelos
    
    Args:
        uf : str pra saber onde ir pegar os csv que vai abrir
    '''
    df_m = pd.read_csv('../dados_versao_final/meteorologicos/'+uf+'.csv',index_col='data',parse_dates=True)
    df_d = pd.read_csv('../dados_versao_final/doencas/'+uf+'.csv')
    
    # Fazendo rolling mean de todas as features, vamo ve    
    # for c in df_m.columns:
    #     df_m[c+'_rm'] = df_m[c].rolling(window=4,min_periods=1).mean()
    
    # Caso va usar o trends
    df_t = pd.read_csv('../dados_versao_final/trends/'+uf+'.csv',index_col='Week',parse_dates=True)
    df_m['trends'] = df_t['buscas']
    
    # Retirando os atributos que deram como menos importantes
    # df_m = df_m.drop([df_m.columns[0],df_m.columns[5],df_m.columns[7]],axis=1)
    
    df_m = df_m.drop('umid_max', axis=1)

    y = df_d['casos'].values
    X = df_m[:].values
    
    return df_m, df_d, X, y

def prepareData(WS,train_size,X,y):
    '''
    Prepara o dataset para o formato que o LSTM recebe como entrada
    
    Args:
        WS : int, tamanho do lookback
        train_size : int, indice para corte entre treino e teste
        X : array, atributos exceto o alvo
        y : array, atributo alvo
    '''
    # Montagem para o formato do modelo
    X = np.array([X[i-WS:i] for i in range(WS, len(X))]) # Original range(WS, len(X))
    y = y[WS:]

    # Separacao em treino/teste
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]
    
    X_train, X_test = X[:train_size], X[-52:]
    y_train, y_test = y[:train_size], y[-52:]
    
    return X_train, X_test, y_train, y_test

def scaleData(X,y,scaler):
    '''
    Aplica transformacao nos dados
    
    Args:
        X : array, dados
        y : array, dados
        scaler : o scaler pra ser usado
    '''
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y
 
def inverseTransform(data,scaler):
    '''
    Desfaz transformacao nos dados
    
    Args:
        X : array, dados
        y : array, dados
        scaler : o scaler pra ser usado
    '''
    return scaler.inverse_transform(data)

#==============================================================================
# SELECAO DE FEATURES
#==============================================================================

# Pra ver se alguma feature pode ser cortada - infelizmente deu que nao
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
def selectFeatures():
    df_m, df_d, X, y = openAndSplit('RN')
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    rfecv = RFECV(estimator=xgboost_model, step=1, cv=tscv, scoring='neg_mean_squared_error')
    
    rfecv.fit(X, y)
    print("Optimal number of features: %d" % rfecv.n_features_)
    
    
    selected_features = np.where(rfecv.support_)[0]
    print("Selected features indices:", selected_features)

#==============================================================================
# METRICAS
#==============================================================================


def calcMetricas(y_true, y_pred):
    '''
    Calcula metricas para avaliar a predicao
    
    Args:
        y_test : array, os valores reais
        y_pred : array, os valores preditos
    '''
    r2 = round(r2_score(y_true, y_pred),3)
    mse = round(mean_squared_error(y_true, y_pred),3)
    mae = round(mean_absolute_error(y_true, y_pred),3)
    mape = round(mean_absolute_percentage_error(y_true, y_pred),3)
    
    return r2, mape, mse, mae


#==============================================================================
# GRAFICOS
#==============================================================================

# Casos total
def fullGraph(df_m,df_d,pred=False,y_pred=[],save=False,uf = '',gambiarra=False,WS=26):
    '''
    Cria grafico comparando predicoes com o real, contrastando com o todo do dataset
    
    Args:
        df_m : dataframe de dados climaticos, vem aqui só pra usar o índice
        df_d : dataframe dos dados do sinan
        pred : bool, se 1 é pq veio junto predicoes pra incluir no grafico
        y_pred : array com valores previstos
        save : bool, se 1 salva a imagem, se 0 apenas exibe ela
        uf : str com a sigla da UF
    '''
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.plot(df_m.index,df_d['casos'], label='Reais')
    if pred:
        if not gambiarra:
            ax.plot(df_m.index[-y_pred.shape[0]:], y_pred, label='Previstos')
        else:
            ax.plot(df_m.index[WS:len(df_m.index)-52], y_pred, label='Previstos')
    #ax.plot(df_m.index,y_pred)
    ax.set_ylabel('Casos')
    ax.set_title('Casos por semana - '+uf)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(fname = './TESTE/'+uf+'_full.png',format='png')
    plt.close(fig)

# Só a previsão
def partialGraph(y_true,y_pred,y_index,uf,save=False):
    '''
    Cria grafico comparando predicoes com o real
    
    Args:
        y_true : array contendo valores reais
        y_pred : array contendo valores preditos
        y_index : array com o índice pro eixo x
        uf : str com a sigla da UF
        save : bool, se 1 salva a imagem, se 0 apenas exibe ela
    '''
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.plot(y_index,y_true, label='Reais')
    ax.plot(y_index,y_pred, label='Previstos')
    ax.grid(True)
    ax.set_title(uf+' - LSTM',fontsize=13)
    ax.legend()
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(fname = './TESTE/'+uf+'_partial.png',format='png')
    plt.close(fig)


def boxplot(metricas,ensemble=False):
    '''
    Cria boxplot das metricas
    
    Args:
        metricas (dataframe) : df contendo as metricas
    '''
    plt.ioff()
    # metricas = pd.read_csv('./TESTE/metricas.csv').drop(['estado','parametros'],axis=1)
    plt.figure(figsize=(10,6))

    plt.subplot(1, 3, 2)
    plt.boxplot(metricas['MSE'])
    plt.title('MSE')
    
    plt.subplot(1, 3, 1)
    plt.boxplot(metricas['MAE'])
    plt.title('MAE')

    plt.subplot(1, 3, 3)
    plt.boxplot(metricas['MAPE'])
    plt.title('MAPE')
    
    if ensemble:
        plt.suptitle('Metricas LSTM - ensemble',fontsize=18)
    else:
        plt.suptitle('Metricas LSTM',fontsize=18)

    plt.tight_layout()

    # plt.show()
    plt.savefig(fname='./TESTE/_boxplot.png')

# metricas = pd.read_csv('./TESTE/metricas.csv')
# boxplot(metricas,ensemble=False)

#==============================================================================
# MONTADNOD OS CSV DE TODAS METRICA
#==============================================================================
# import numpy as np
# milho = np.mean(rf['MAPE'])

# silbo = pd.DataFrame()
# silbo['e'] = lstm['estado']
# silbo['MAPE_rf'] = rf['MAPE']
# silbo['R2_rf'] = rf['R2']
# silbo['MAPE_svm'] = svr['MAPE']
# silbo['R2_svm'] = svr['R2']
# silbo['MAPE_lstm'] = lstm['MAPE']
# silbo['R2_lstm'] = lstm['R2']
# silbo.to_csv('MAPE_R2.csv',index=True)

# silbo = pd.DataFrame()

# silbo= pd.read_csv('MAE_MSE.csv',index_col='e')
# silbo = silbo.sort_index()

#==============================================================================
# FAZENDO BOXPLOT DE METRICAS COMPARANDO
#==============================================================================

# rf = pd.read_csv('_rf_semtrends.csv')
# svr = pd.read_csv('_svr_semtrends.csv')
# lstm = pd.read_csv('_lstm_semtrends.csv')

# rf = pd.read_csv('rf.csv')
# svr = pd.read_csv('svr.csv')
# rf2 = pd.read_csv('_metricas_escala_original.csv')
# lstm = pd.read_csv('lstm.csv')

# # from sklearn.preprocessing import MinMaxScaler
# # import numpy as np
# # scaler = MinMaxScaler()
# # f = np.array(lstm2['MAE'])
# # teste = scaler.fit_transform(silbo)

# l = [rf['MAPE'],svr['MAPE'],lstm['MAPE']]
# # # l = [silbo['MSE_rf'],silbo['MSE_svm'],silbo['MSE_lstm']]
# # # a = [rf['MSE'],svr['MSE'],lstm['MSE']]
# labels = ['RF','SVR','LSTM']
# meanlineprops = dict(linewidth=2.5)

# import numpy as np

# np.mean(l[2])

# # PRA DOIS GRAFICO
# # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)
# # axs[0].boxplot(l, notch=False, meanprops=meanlineprops, showmeans=True, meanline=True,labels=labels)
# # axs[0].set_title('MAPE')
# # axs[0].xticks([1, 2, 3], ['RF','SVR','LSTM'], fontsize=12)
# # axs[1].boxplot(a, notch=False, meanprops=meanlineprops, showmeans=True, meanline=True,labels=labels)
# # axs[1].set_title('R2')
# # axs[1].xticks([1, 2, 3], ['RF','SVR','LSTM'], fontsize=12)
# # import numpy as np
# # print(np.mean(silbo['MAE_lstm']))

# # PRA UM SO
# plt.figure(figsize=(6,6))
# plt.boxplot(l, notch=False, meanprops=meanlineprops, showmeans=True, meanline=True)
# plt.xticks([1, 2, 3], labels, fontsize=12)

# # EXIBIR O GRAFIOC
# plt.title('Performance MAPE',fontsize=13)
# plt.show()



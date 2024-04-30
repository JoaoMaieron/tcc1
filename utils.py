import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

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
# df = pd.read_csv('../dados_versao_final/meteorologicos/SC.csv',index_col='data',parse_dates=True)
# for c in df.columns:
#     df[c+'_rm'] = df[c].rolling(window=8,min_periods=1).std()
# df = df.dropna()

def openAndSplit(uf):
    '''
    Abre e prepara dados pra alimentar pros preditores
    
    Args:
        uf : str pra saber onde ir pegar os csv que vai abrir
    '''
    
    df_m = pd.read_csv('../dados_versao_final/meteorologicos/'+uf+'.csv',index_col='data',parse_dates=True)
    df_d = pd.read_csv('../dados_versao_final/doencas/'+uf+'.csv')
    
    # Caso va usar o trends
    df_t = pd.read_csv('../dados_versao_final/trends/'+uf+'.csv',index_col='Week',parse_dates=True)
    df_m['trends'] = df_t['buscas']
    
    # Fazendo rolling de todas as features, vamo ve
    # df_m2 = pd.DataFrame()
    # for c in df_m.columns:
    #     df_m2[c] = df_m[c].rolling(window=26,min_periods=1).mean()
    #     # df_m2[c+'_std'] = df_m[c].rolling(window=26,min_periods=1).std()    
    # #     df_m[c+'_median'] = df_m[c].rolling(window=8,min_periods=1).median()
    # df_m2.fillna(0,inplace=True)
    # y = df_d['casos'].values
    # X = df_m2[:].values
    # return df_m2, df_d, X, y

    df_m = df_m.drop('umid_max', axis=1)

    y = df_d['casos'].values
    X = df_m[:].values
    
    return df_m, df_d, X, y

# df_m, df_d, X, y = openAndSplit('AP')

import seaborn as sns
def pearsonMatrix(uf):
    '''
    Faz a matriz de Pearson do dataset climatico
    
    Args:
        uf : str, sigla do estado
    '''
    plt.ioff()
    df_m, df_d, X, y = openAndSplit(uf)
    matriz_correlacao = df_m.corr(method='pearson')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlação (Pearson) - "+uf)
    # plt.show()
    plt.savefig(fname = './TESTE/'+uf+'_pearson.png',format='png')
    plt.close()

# for r in regioes:
#     for uf in r:
#         pearsonMatrix(uf)

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
def fullGraph(df_m,df_d,pred=False,y_pred=[],save=False,uf = '',gambiarra=False):
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
            ax.plot(df_m.index[:y_pred.shape[0]], y_pred, label='Previstos')
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
    ax.set_title(uf + ' - SVR',fontsize=13)
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
        plt.suptitle('Metricas SVR - ensemble',fontsize=18)
    else:
        plt.suptitle('Metricas SVR',fontsize=18)

    plt.tight_layout()

    # plt.show()
    plt.savefig(fname='./TESTE/_boxplot.png')

# metricas = pd.read_csv('./TESTE/metricas.csv')
# boxplot(metricas,ensemble=True)

# def apply_variance_threshold(data, threshold=0.01):
#     # Separate features and target variable (assuming target is in the last column)
#     X = data.iloc[:, :]
#     # Apply variance threshold
#     selector = VarianceThreshold(threshold=threshold)
#     X_selected = selector.fit_transform(X)

#     # Get the indices of selected features
#     selected_feature_indices = selector.get_support(indices=True)
    
#     print(selected_feature_indices)

#     # Combine selected features and target variable
#     # selected_data = pd.concat([pd.DataFrame(X_selected)], axis=1)

#     return selected_feature_indices
# def abcd():
#     common_feature_indices = set()
#     for uf in UFCodes.values():
#         df_m, df_d, X, y = utils.openAndSplit(uf)
        
#         print(uf,end=' ')
#         selected_indices = apply_variance_threshold(df_m)
        
#         common_feature_indices.update(selected_indices)
#     # Convert the set of common feature indices to a sorted list
#     common_feature_indices = sorted(list(common_feature_indices))

#     # Print the common feature indices
#     # print("Common Feature Indices:", common_feature_indices)
# abcd()

# for uf in UFCodes.values():
#     df_m, df_d, X, y = utils.openAndSplit(uf)
#     selector = VarianceThreshold(threshold=0.01)
#     X_selected = selector.fit_transform(X)
#     print(X_selected.shape)

# from sklearn.preprocessing import MinMaxScaler

# df = pd.read_csv('../dados_versao_final/trends/CE.csv',index_col='Week',parse_dates=True)
# df_d = pd.read_csv('../dados_versao_final/doencas/CE.csv')

# scaler = MinMaxScaler()
# t = df['buscas'].values
# t = scaler.fit_transform(t.reshape(-1,1))
# d = df_d['casos'].values
# d = scaler.fit_transform(d.reshape(-1,1))

# fig, ax = plt.subplots(figsize=(10,6))
# ax.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.plot(df.index,t, label='Trends')
# ax.plot(df.index,d, label='Casos de dengue')

#ax.plot(df_m.index,y_pred)
# ax.set_ylabel('Casos')
# ax.set_title('Comparativo Trends x casos de dengue - CE')
# ax.grid(True)
# ax.legend()
# plt.tight_layout()
# plt.show()
# if not save:
#     plt.show()
# else:
#     plt.savefig(fname = './TESTE/'+uf+'_full.png',format='png')
# plt.close(fig)
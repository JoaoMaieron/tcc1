import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#==============================================================================
# PRE PROCESS
#==============================================================================

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


#==============================================================================
# GRAFICOS
#==============================================================================

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
        plt.savefig(fname = './TESTE/'+uf+'_full.png',format='png')

# Só a previsão
def partialGraph(y_true,y_pred,y_index,uf,save=False):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.plot(y_index,y_true, label='Reais')
    ax.plot(y_index,y_pred, label='Previstos')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(fname = './TESTE/'+uf+'_partial.png',format='png')






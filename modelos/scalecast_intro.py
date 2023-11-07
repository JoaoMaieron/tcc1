# -*- coding: utf-8 -*-

#======================================================================================================
# BIBLIOTECAS QUE SEMPRE PRECISA
#======================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
from scalecast.MVForecaster import MVForecaster
from scalecast.Pipeline import Pipeline, Transformer, Reverter
from scalecast.Pipeline import MVPipeline
from scalecast.util import find_optimal_transformation
from scalecast.SeriesTransformer import SeriesTransformer
from scalecast.util import gen_rnn_grid
from tensorflow.keras.callbacks import EarlyStopping
from scalecast import GridGenerator

casos_mg = pd.read_csv('../dados_versao_final/doencas/MG.csv')
tempo_mg = pd.read_csv('../dados_versao_final/meteorologicos/MG.csv',parse_dates=True, index_col='data')

import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(10,6))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(tempo_mg.index,casos_mg['casos'])
ax.set_ylabel('Casos')
ax.set_title('Casos de dengue relatados, por semana')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# Instanciando um forecaster para cada var
def createForecaster(y,current_dates,future_dates,estimator):
    return Forecaster(y=y, current_dates=current_dates, future_dates=future_dates,estimator=estimator)

future_dates = 52
estimator = 'lstm'
test_length = .15


f_casos = createForecaster(casos_mg['casos'], tempo_mg.index, future_dates,estimator)
f_tmed = createForecaster(tempo_mg['temp_media'], tempo_mg.index, future_dates,estimator)
f_umid = createForecaster(tempo_mg['umid_media'], tempo_mg.index, future_dates,estimator)

f_casos.seasonal_decompose().plot()
plt.show()

f_casos.plot()
plt.show()

transformer = SeriesTransformer(f_casos)
f_casos = transformer.MinMaxTransform(train_only=True)

mvf = MVForecaster(
    f_casos,
    f_tmed,
    f_umid,
    test_length=test_length,
    metrics=['rmse','r2']
    )










GridGenerator.get_mv_grids(overwrite=True)

# PRINCIPAL
f_casos = Forecaster(y=casos_mg['casos'],current_dates=tempo_mg.index,future_dates=104,test_length=.15)

# OS AMIGUES
f_tmed = Forecaster(y=tempo_mg['temp_media'],current_dates=tempo_mg.index,future_dates=104,test_length=.15)
# f_tmax = Forecaster(y=tempo_mg['temp_max'],current_dates=tempo_mg.index,future_dates=104,test_length=.15)
# f_tmin = Forecaster(y=tempo_mg['temp_min'],current_dates=tempo_mg.index,future_dates=104,test_length=.15)
# f_prec = Forecaster(y=tempo_mg['prec'],current_dates=tempo_mg.index,future_dates=104,test_length=.15)
f_umid = Forecaster(y=tempo_mg['umid_media'],current_dates=tempo_mg.index,future_dates=104,test_length=.15)
# f_vento = Forecaster(y=tempo_mg['vento'],current_dates=tempo_mg.index,future_dates=104,test_length=.15)
# var_list = [f_tmed,f_tmax,f_tmin,f_prec,f_umid,f_vento]

# SEASONAL REGRESSORS ESTAO AI PARA CAPTURAR PADROES SAZONAIS
def add_vars(f,**kwargs):
    f.add_seasonal_regressors(
        'month',
        'quarter',
        'week',
        raw=False,
        sincos=True
    )
def mvforecaster2(mvf,models):
    mvf.tune_test_forecast(
        models,
        cross_validate=True,
        k=2,
        rolling=True,
        dynamic_tuning=104,
        dynamic_testing=104,
        limit_grid_size = .2,
        error = 'warn',
    )

t_casos, r_casos = find_optimal_transformation(
    f_casos,
    lags = 104,
    m = 52,
    estimator = 'lstm',
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)
t_tmed, r_tmed = find_optimal_transformation(
    f_tmed,
    lags = 104,
    m = 52,
    estimator = 'lstm',
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)
t_tmax, r_tmax = find_optimal_transformation(
    f_tmax,
    lags = 104,
    m = 52,
    estimator = 'lstm',
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)
t_tmin, r_tmin = find_optimal_transformation(
    f_tmin,
    lags = 104,
    m = 52,
    estimator = 'lstm',
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)
t_prec, r_prec = find_optimal_transformation(
    f_prec,
    lags = 104,
    m = 52,
    estimator = 'lstm',
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)
t_umid, r_umid = find_optimal_transformation(
    f_umid,
    lags = 104,
    m = 52,
    estimator = 'lstm',
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)
t_vento, r_vento = find_optimal_transformation(
    f_vento,
    lags = 104,
    m = 52,
    estimator = 'lstm',
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)

mvpipeline = MVPipeline(
    steps = [
        ('Transform',[t_casos,t_tmed,t_tmax,t_tmin,t_prec,t_umid,t_vento]),
        ('Add Xvars',[add_vars]*7),
        ('Forecast',[mvforecaster2]*7,),
        ('Revert',[r_casos,r_tmed,r_tmax,r_tmin,r_prec,r_umid,r_vento]),
    ],
    test_length = 20,
    cis = True,
    names = ['casos','t_media','t_max','t_min','prec','umid','vento'],    
    verbose=True
)

f_casos, f_tmed, f_tmax, f_tmin, f_prec, f_umid, f_vento = mvpipeline.fit_predict(
    f_casos, 
    f_tmed, 
    f_tmax, 
    f_tmin, 
    f_prec, 
    f_umid, 
    f_vento,
    models=[
        'lstm','lstm','lstm','lstm','lstm','lstm','lstm'
    ],
) # returns a tuple of Forecaster objects

# ===================================================================
f_casos.plot_test_set(order_by='TestSetRMSE',ci=True)
plt.show()


#======================================================================================================
# UNIVARIATE
#======================================================================================================
# Fonte: https://scalecast-examples.readthedocs.io/en/latest/index.html
data = pd.read_csv('avocado.csv',parse_dates=True)
data = data.drop('Unnamed: 0',axis=1)
data = data.sort_values(['region','type','Date'])

# Instanciando Forecaster
# https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html
from scalecast.Forecaster import Forecaster
volume = data.groupby('Date')['Total Volume'].sum()
f = Forecaster(y= volume, current_dates=volume.index, future_dates=13)
print(f)

#=====================================================
# Graficos para determinar os parametros do Forecaster
#=====================================================
f.plot()
plt.show()
# Decomposicao dos dados
plt.rc("figure",figsize=(10,6))
f.seasonal_decompose().plot()
plt.show()
# Graficos ACF e PACF
figs, axs = plt.subplots(2, 1,figsize=(9,9))
f.plot_acf(ax=axs[0],title='ACF',lags=26,color='black')
f.plot_pacf(ax=axs[1],title='PACF',lags=26,color='#B2C248',method='ywm')
plt.show()

#=====================================================
# Parametrizando o forecaster
#=====================================================

# Tamanho do teste em % com maxima 1; quando 0, nao é feito teste
f.set_test_length(.15) # last 15% of observed values
# Avaliar intervalos de confianca - nao funciona se 0 acima
f.eval_cis(
    mode = True, # tell the object to evaluate intervals
    cilevel = .95, # 95% confidence level
)

# Adicionando mais parametros ao forecaster
f.add_time_trend()
f.add_seasonal_regressors('week',raw=False,sincos=True)
f.add_ar_terms(13)

#=====================================================
# Definicao dos modelos e primeiros resultados
#=====================================================
# Lista: https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html
# Vao ser feitas previsoes em 13 passos

# Modelos lineares do sklearn
f.set_estimator('mlr')
f.manual_forecast(dynamic_testing=13)
f.set_estimator('lasso')
f.manual_forecast(alpha=0.2,dynamic_testing=13)
f.set_estimator('ridge')
f.manual_forecast(alpha=0.2,dynamic_testing=13)
f.set_estimator('elasticnet')
f.manual_forecast(alpha=0.2,l1_ratio=0.5,dynamic_testing=13)
f.set_estimator('sgd')
f.manual_forecast(alpha=0.2,l1_ratio=0.5,dynamic_testing=13)
# Graficos dos resultados
f.plot_test_set(ci=True,models=['mlr','lasso','ridge','elasticnet','sgd'],order_by='TestSetRMSE')
plt.show()
f.plot(ci=True,models=['mlr','lasso','ridge','elasticnet','sgd'],order_by='TestSetRMSE')
plt.show()

# Modelos nao lineares do sklearn
f.set_estimator('rf')
f.manual_forecast(max_depth=2,dynamic_testing=13)
f.set_estimator('gbt')
f.manual_forecast(max_depth=2,dynamic_testing=13)
f.set_estimator('xgboost')
f.manual_forecast(gamma=1,dynamic_testing=13)
f.set_estimator('lightgbm')
f.manual_forecast(max_depth=2,dynamic_testing=13)
f.set_estimator('catboost')
f.manual_forecast(depth=4,verbose=False,dynamic_testing=13)
f.set_estimator('knn')
f.manual_forecast(n_neighbors=5,dynamic_testing=13)
f.set_estimator('mlp')
f.manual_forecast(hidden_layer_sizes=(50,50),solver='lbfgs',dynamic_testing=13)
#Graficos dos resultados
f.plot_test_set(
    ci=True,
    models=['rf','gbt','xgboost','lightgbm','catboost','knn','mlp'],
    order_by='TestSetRMSE'
)
plt.show()
f.plot(ci=True,models=['rf','gbt','xgboost','lightgbm','knn','mlp'],order_by='TestSetRMSE')
plt.show()

# Modelo ARIMA
# DEMOROU MUITO - NAO UTILIZAR ESSE 
from scalecast.auxmodels import auto_arima
auto_arima(f,m=52)
f.plot_test_set(models='auto_arima',ci=True)
plt.show()
f.plot(models='auto_arima',ci=True)
plt.show()

#======================================================================================================
# MULTIVARIATE
#======================================================================================================

price = data.groupby('Date')['AveragePrice'].mean()

# Instanciando forecasters
fvol = Forecaster(y=volume,current_dates=volume.index,future_dates=13)
fprice = Forecaster(y=price,current_dates=price.index,future_dates=13)

fvol.add_time_trend()
fvol.add_seasonal_regressors('week',raw=False,sincos=True)

# Objeto usado para multi, em vez do forecaster normal
from scalecast.MVForecaster import MVForecaster
mvf = MVForecaster(
    fvol,
    fprice,
    merge_Xvars='union',
    names=['volume','price'],
)
print(mvf)

# Analise grafica dos dados
mvf.plot(series='volume')
plt.show()
mvf.plot(series='price')
plt.show()

#=============================================================
# Heat maps das variaveis 
#=============================================================
mvf.corr(disp='heatmap',cmap='Spectral',annot=True,vmin=-1,vmax=1)
plt.show()
mvf.corr_lags(y='price',x='volume',disp='heatmap',cmap='Spectral',annot=True,vmin=-1,vmax=1,lags=13)
plt.show()
mvf.corr_lags(y='volume',x='price',disp='heatmap',cmap='Spectral',annot=True,vmin=-1,vmax=1,lags=13)
plt.show()

# Parametrizando o multi forecaster
mvf.set_test_length(.15)
#mvf.set_optimize_on('volume') # we care more about predicting volume and price is just used to make those predictions more accurate
# by default, the optimizer uses an average scoring of all series in the MVForecaster object
mvf.eval_cis() # tell object to evaluate cis
print(mvf)

#=============================================================
# Rodando modelos
#=============================================================
mvf.set_estimator('elasticnet')
mvf.manual_forecast(alpha=0.2,dynamic_testing=13,lags=13)

mvf.set_estimator('xgboost')
mvf.manual_forecast(gamma=1,dynamic_testing=13,lags=13)

from scalecast.auxmodels import mlp_stack
mvf.export('model_summaries')

# Determine best deve ser dentre:
# ['TestSetRMSE', 'TestSetMAPE', 'TestSetMAE', 'TestSetR2', 'InSampleRMSE', 
# 'InSampleMAPE', 'InSampleMAE', 'InSampleR2', 'ValidationMetricValue']
mlp_stack(mvf,model_nicknames=['elasticnet','xgboost'],lags=13)
mvf.set_best_model(determine_best_by='TestSetRMSE')
# Resultados
mvf.plot_test_set(ci=True,series='volume',put_best_on_top=True)
plt.show()
mvf.plot(ci=True,series='volume',put_best_on_top=True)
plt.show()

#=============================================================
# Transformacoes
#=============================================================
# https://scalecast-examples.readthedocs.io/en/latest/misc/introduction/Introduction2.html#Transformations
from scalecast.SeriesTransformer import SeriesTransformer
f_trans = Forecaster(y=volume,current_dates=volume.index,future_dates=13)
f_trans.set_test_length(.15)
f_trans.set_validation_length(13)
transformer = SeriesTransformer(f_trans)
# Transformacoes podem todas ser revertidas

# Differencing
f_trans = transformer.DiffTransform(1)
f_trans = transformer.DiffTransform(52)
f_trans = transformer.DetrendTransform()
f_trans.plot()
plt.show()

f_trans.add_time_trend()
f_trans.add_seasonal_regressors('week',sincos=True,raw=False)
f_trans.add_ar_terms(13)
f_trans.set_estimator('xgboost')
f_trans.manual_forecast(gamma=1,dynamic_testing=13)
# Resultados
f_trans.plot_test_set()
plt.show()
f_trans.plot()
plt.show()

# Reversao é feita so chamando elas na ordem oposta a de antes
f_trans = transformer.DetrendRevert()
f_trans = transformer.DiffRevert(52)
f_trans = transformer.DiffRevert(1)
f_trans.plot_test_set()
plt.show()
f_trans.plot()
plt.show()

# MinMax
f_trans = transformer.MinMaxTransform(train_only=True)
f_trans.plot()
plt.show()
# Usando os mesmos parametros do anterior ficou muito ruim a previsao
f_trans.add_time_trend()
f_trans.add_seasonal_regressors('week',sincos=True,raw=False)
f_trans.add_ar_terms(13)
f_trans.set_estimator('xgboost')
f_trans.manual_forecast(gamma=1,dynamic_testing=13)
f_trans.plot_test_set()
plt.show()
f_trans.plot()
plt.show()

#======================================================================================================
# PIPELINES
#======================================================================================================
# These are objects similar to scikit-learn pipelines that offer readable and 
# streamlined code for transforming, forecasting, and reverting.
# https://scalecast.readthedocs.io/en/latest/Forecaster/Pipeline.html
from scalecast.Pipeline import Transformer, Reverter, Pipeline, MVPipeline

# Instanciando parametros do pipeline
# Um de cada transformer, reverser, forecaster
f_pipe = Forecaster(y=volume,current_dates=volume.index,future_dates=13)
f_pipe.set_test_length(.15)

def forecaster(f):
    f.add_time_trend()
    f.add_seasonal_regressors('week',raw=False,sincos=True)
    f.add_ar_terms(13)
    f.set_estimator('lightgbm')
    f.manual_forecast(max_depth=2)

transformer = Transformer(
    transformers = [
        ('DiffTransform',1),
        ('DiffTransform',52),
        ('DetrendTransform',)
    ]
)

reverter = Reverter(
    reverters = [
        ('DetrendRevert',),
        ('DiffRevert',52),
        ('DiffRevert',1)
    ],
    base_transformer = transformer,
)

# Rodando o pipeline
pipeline = Pipeline(
    steps = [
        ('Transform',transformer),
        ('Forecast',forecaster),
        ('Revert',reverter),
    ]
)

f_pipe = pipeline.fit_predict(f_pipe)
# Resultados
f_pipe.plot_test_set()
plt.show()
f_pipe.plot()
plt.show()

#======================================================================================================
# MAIS EXEMPLOS DE PIPELINE
#======================================================================================================

# Univariate
transformer = Transformer(transformers = [('DetrendTransform',{'poly_order':2}),'DeseasonTransform'])
reverter = Reverter(reverters=['DeseasonRevert','DetrendRevert'], base_transformer=transformer)
# For this example, we will use 18 lags, one layer, a tanh activation function, and 200 epochs
def forecaster(f):
    f.set_estimator('rnn')
    f.manual_forecast(
        lags = 18,
        layers_struct = [
            ('LSTM',{'units':36,'activation':'tanh'}),
        ],
        epochs=200,
        plot_loss = True,
        call_me = 'lstm'
    )
# Combina-se tudo em um pipeline
pipeline = Pipeline(
    steps=[
        ('Transform',transformer),
        ('Forecast',forecaster),
        ('Revert',reverter)
        ]
    )
# Rodando e exibindo previsao
f = pipeline.fit_predict(f)
f.plot()
plt.show()

# Multivariate
data = pd.read_csv('avocado.csv',parse_dates=True)
data = data.drop('Unnamed: 0',axis=1)



def forecaster(f):
    f.set_estimator('lstm'),
    f.manual_forecast(
        lags=20,        
        dropout=0.2,
        plot_loss=True        
        )


# EXEMPLO QUE PARECIA BOM MAS NAO FUNCIONOU

# Vamos prever vol com base em price
vol = data.groupby('Date')['Total Volume'].sum()
price = data.groupby('Date')['AveragePrice'].sum()

fvol = Forecaster(
    y=vol,
    current_dates=vol.index,
    test_length = 13,
    validation_length=13,
    future_dates=13,
    metrics = ['rmse','r2'],
    )
fvol.plot()
plt.show()

# Isso aqui simplesmente te da a melhor transformacao
from scalecast.util import find_optimal_transformation
transformer, reverter = find_optimal_transformation(
    fvol,
    set_aside_test_set=True, # prevents leakage so we can benchmark the resulting models fairly
    return_train_only = True, # prevents leakage so we can benchmark the resulting models fairly
    verbose=True,
    detrend_kwargs=[
        {'loess':True},
        {'poly_order':1},
        {'ln_trend':True},
    ],
    m = 52,
    test_length = 4,
)

# Isso aqui da os melhores hiperparametros
# Estamos aqui pegando uma univariate pra testar contrastando com a multi
from scalecast.util import gen_rnn_grid
from tensorflow.keras.callbacks import EarlyStopping
rnn_grid = gen_rnn_grid(
    layer_tries = 10,
    min_layer_size = 3,
    max_layer_size = 5,
    units_pool = [100], # Default [8,16,32,64] suponho que seja o batch size?
    epochs = [25,50],
    dropout_pool = [0,0.05], # Possiveis valores de dropout
    callbacks=EarlyStopping(
      monitor='val_loss',
      patience=3,
    ),
    random_seed = 20,
) # creates a grid of hyperparameter values to tune the LSTM model

# Agora o fit da univariate
fvol.add_ar_terms(13) # the model will use 13 series lags
fvol.set_estimator('rnn')
fvol.ingest_grid(rnn_grid)
fvol.tune() # uses a 13-period validation set
fvol.auto_forecast(call_me='lstm_univariate')

# A seguir, para chegar ao multi, aplicamos as mesmas transformacoes aos outros dados
fprice = Forecaster(
    y = price,
    current_dates = price.index,
    future_dates = 13,
)
fprice = transformer.fit_transform(fprice)

fvol.add_series(fprice.y,called='price')
fvol.add_lagged_terms('price',lags=13,drop=True)
fvol.ingest_grid(rnn_grid)
fvol.tune()
fvol.auto_forecast(call_me='lstm_multivariate')


fvol.set_estimator('naive')
fvol.manual_forecast()

# Reverte a transformacao na saida e plota os resultados
fvol = reverter.fit_transform(fvol)
fvol.plot_test_set(order_by='TestSetRMSE')
plt.show()


# ESSE AQUI AO MENOS RODOU - FVOL SAO OS CASOS, FPRICE SAO AS TEMP MEDIAS

fvol = Forecaster(
    y = casos_mg['casos'],
    current_dates = tempo_mg.index,
    test_length = 13,
    validation_length = 13,
    future_dates = 13,
    metrics = ['rmse','r2'],
)
fprice = Forecaster(
    y = tempo_mg['temp_media'],
    current_dates = tempo_mg.index,
    future_dates = 13,
)

transformer, reverter = find_optimal_transformation(
    fvol,
    set_aside_test_set=True, # prevents leakage so we can benchmark the resulting models fairly
    return_train_only = True, # prevents leakage so we can benchmark the resulting models fairly
    verbose=True,
    detrend_kwargs=[
        {'loess':True},
        {'poly_order':1},
        {'ln_trend':True},
    ],
    m = 52, # what makes one seasonal cycle?
    test_length = 4,
)
fprice = transformer.fit_transform(fprice)
fvol = transformer.fit_transform(fvol)

rnn_grid = gen_rnn_grid(
    layer_tries = 10,
    min_layer_size = 3,
    max_layer_size = 5,
    units_pool = [100],
    epochs = [100],
    dropout_pool = [0,0.05],
    validation_split=.2,
    callbacks=EarlyStopping(
      monitor='val_loss',
      patience=3,
    ),
    random_seed = 20,
) # creates a grid of hyperparameter values to tune the LSTM model
def forecaster(fvol,fprice):
    # naive forecast for benchmarking
    fvol.set_estimator('naive')
    fvol.manual_forecast()
    # univariate lstm model
    fvol.add_ar_terms(13) # the model will use 13 series lags
    fvol.set_estimator('rnn')
    fvol.ingest_grid(rnn_grid)
    fvol.tune()
    fvol.auto_forecast(call_me='lstm_univariate')
    # multivariate lstm model
    fvol.add_series(fprice.y,called='price')
    fvol.add_lagged_terms('price',lags=13,drop=True)
    fvol.ingest_grid(rnn_grid)
    fvol.tune()
    fvol.auto_forecast(call_me='lstm_multivariate')
forecaster(fvol=fvol,fprice=fprice)

fvol = reverter.fit_transform(fvol)
fvol.plot_test_set(order_by='TestSetRMSE')
plt.savefig('MEU PAU.png')
plt.show()



# EXEMPLO DE PIPELINE AUTOMATICO - UNIVARIAVEL


GridGenerator.get_example_grids(overwrite=True)

f_pipe_aut = Forecaster(y=casos_mg['casos'],current_dates=tempo_mg.index,future_dates=104)
f_pipe_aut.set_test_length(.15)

def forecaster_aut(f,models):
    f.auto_Xvar_select(
        estimator='elasticnet',
        monitor='TestSetMAE',
        alpha=0.2,
        irr_cycles = [26],
    )
    f.tune_test_forecast(
        models,
        cross_validate=True,
        k=3,
        
        dynamic_tuning=104,
        dynamic_testing=104,
    )
    f.set_estimator('combo')
    f.manual_forecast()

transformer_aut, reverter_aut = find_optimal_transformation(
    f_pipe_aut,
    lags = 104,
    m = 52,
    monitor = 'mae',
    estimator = 'elasticnet',
    alpha = 0.2,
    test_length = 104,
    num_test_sets = 3,
    space_between_sets = 4,
    verbose = True,
)

pipeline_aut = Pipeline(
    steps = [
        ('Transform',transformer_aut),
        ('Forecast',forecaster_aut),
        ('Revert',reverter_aut),
    ]
)

f_pipe_aut = pipeline_aut.fit_predict(
    f_pipe_aut,
    models=[
        'mlr',
        'elasticnet',
        'xgboost',
        'lightgbm',
        'knn',
    ],
)

print(f_pipe_aut)
f_pipe_aut.plot_test_set(order_by='TestSetRMSE')
plt.show()
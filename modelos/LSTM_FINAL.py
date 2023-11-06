# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_m = pd.read_csv('../dados_versao_final/meteorologicos/MG.csv',index_col='data',parse_dates=True)
df_m = df_m.drop('2023-01-01')
#df_m['umid_media'] = df_m['umid_media'].abs() # Alguns valores estavam negativos, deixa quieto
df_d = pd.read_csv('../dados_versao_final/doencas/MG.csv')
df_d = df_d.drop(0)

# Separacao variaveis
X = df_m[:].values
y = df_d['casos'].values

# Transformacao MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Montagem para o formato do modelo
WS = 26  # HIPERPARAMETRO
X = np.array([X[i-WS:i] for i in range(WS, len(X))])
y = y[WS:]

# Separacao em treino/teste
train_size = int(0.8 * len(X))
#train_size = 626
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]





from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

# Camadas do modelo segundo artigo
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
model = Sequential()
model.add(LSTM(units=64, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 64, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 64, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 64, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 64))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

#=================================================================================
# SEM OTIMIZACAO
#=================================================================================
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=Adam(learning_rate=0.001), loss = 'mean_squared_error',metrics=["accuracy", "mape", "mse"])

model.fit(X_train,y_train, epochs = 200, batch_size = 64)

from scikeras.wrappers import KerasRegressor
best_model = KerasRegressor(
    model=model,
# 	build_fn=<keras.src.engine.sequential.Sequential object at 0x798858ed32e0>,
	warm_start=False,
	random_state=None,
	optimizer=RMSprop(learning_rate=0.001),
	loss='mean_squared_error',
	metrics=None,
	batch_size=64,
	validation_batch_size=None,
	verbose=True,
	callbacks=None,
	validation_split=0.0,
	shuffle=True,
	run_eagerly=False,
	epochs=200)



# Se vier esquisito, usar .flatten()
y_pred = model.predict(X_test)

# Consegui melhores resultados com epochs=500, batch_size=32, WS=26
best_model.fit(X_train,y_train, epochs = 500, batch_size = 32)
y_pred = best_model.predict(X_test)


# Desfaz transformacoes
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)


#=================================================================================
# COM GRID SEARCH
#=================================================================================
# com learning_rate=0.0001, ws = 26
# {'batch_size': 64, 'epochs': 200}
# best_model:
#     KerasRegressor(
# 	model=None
# 	build_fn=<keras.src.engine.sequential.Sequential object at 0x798858ed32e0>
# 	warm_start=False
# 	random_state=None
# 	optimizer=rmsprop
# 	loss=None
# 	metrics=None
# 	batch_size=64
# 	validation_batch_size=None
# 	verbose=True
# 	callbacks=None
# 	validation_split=0.0
# 	shuffle=True
# 	run_eagerly=False
# 	epochs=200)

model.compile(optimizer='adam', loss='mean_squared_error')

# Grid search de hiperparametrosx
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
lstm_model = KerasRegressor(build_fn=model, verbose=True, epochs=100)
param_grid = {
    'units' : [32, 64, 128],
    'dropout_rate' : [0.2, 0.4, 0.6],
    'batch_size': [16, 32, 64],
    #'epochs': [100, 200, 300],
    # learning_rate: 0.0001 
}
# cv=3 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
grid_search = GridSearchCV(
    #temp_folder = './temp/',
    n_jobs=-1,
    estimator=lstm_model, 
    param_grid=param_grid, 
    scoring='neg_mean_squared_error', # testar 'mean_squared_error', 
    verbose=2,
    cv=3)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Se vier esquisito, usar .flatten()
y_pred = best_model.predict(X_test)

# Desfaz transformacoes
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Caso queira salvar/carregar modelo
from keras.models import load_model
best_model.save('./LSTM_final')
loaded_model = load_model('./LSTM_final')

#=================================================================================
# COM LOOPS
#=================================================================================

def fit_lstm(train, test, raw, scaler, batch_size, nb_epoch, neurons):
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
		# evaluate model on train data
		raw_train = raw[-(len(train)+len(test)+1):-len(test)]
		train_rmse.append(evaluate(model, raw_train, train, scaler, 0, batch_size))
		model.reset_states()
		# evaluate model on test data
		raw_test = raw[-(len(test)+1):]
		test_rmse.append(evaluate(model, raw_test, test, scaler, 0, batch_size))
		model.reset_states()
	history = DataFrame()
	history['train'], history['test'] = train_rmse, test_rmse
	return history


raw = X[:train_size]

repeats = 10
n_batch = 4
n_epochs = 500
n_neurons = 1
# run diagnostic tests
for i in range(repeats):
	history = fit_lstm(X, test_scaled, raw_values, scaler, n_batch, n_epochs, n_neurons)
	pyplot.plot(history['train'], color='blue')
	pyplot.plot(history['test'], color='orange')
	print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))
pyplot.savefig('epochs_diagnostic.png')


#=================================================================================
# AVALIACAO 
#=================================================================================

# Acredito que use as metricas do grid_search
mse = best_model.evaluate(X_test, y_test, verbose=2)
mse = model.evaluate(X_test, y_test, verbose=2)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

#=================================================================================
# GRAFICOS
#=================================================================================

# Casos total
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(10,6))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(df_m.index,df_d['casos'])
ax.plot(df_m.index,y_pred)
ax.set_ylabel('Casos')
ax.set_title('Casos de dengue relatados, por semana')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# Loss function - apos o metodo fit
plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()
plt.plot(range(len(best_model.history.history['loss'])), best_model.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()




# Comparacao final - PRECISA AJUSTAR A ESCALA
plt.plot(y_test, color = 'red', label = 'Numeros reais')
plt.plot(y_pred, color = 'blue', label = 'Numeros previstos')
plt.title('Previsao de casos - MG')
plt.xlabel('Semanas')
plt.ylabel('Nro de casos')
plt.legend()
plt.show()



test = pd.DataFrame({'data' : df_m.index[len(df_m)-WS:], 'pred': y_pred})


import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(10,6))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(df_m.index,df_d['casos'])
ax.plot(df_m.index,y_pred)
ax.set_ylabel('Casos')
ax.set_title('Casos de dengue relatados, por semana')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()



















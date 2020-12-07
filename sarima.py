# grid search sarima hyperparameters for daily female dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from matplotlib import pyplot
from numpy import percentile


# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=True, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress for each test value. Commented for performance reasons
		# print('>expected=%.3f, predicted=%.3f' % (test[i], yhat))
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error, test, predictions

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result, _, _ = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result, _, _ = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models


def plot_sarima(expected, predicted):
	pyplot.plot(expected, label='Valores Esperados')
	pyplot.plot(predicted, label='Valores Encontrados')
	pyplot.legend()
	pyplot.grid()
	pyplot.show()


def remove_outliers(data):
	# calculate interquartile range
	q25, q75 = percentile(data, 25), percentile(data, 75)
	iqr = q75 - q25
	print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
	# calculate the outlier cutoff
	cut_off = iqr * 1.5
	lower, upper = q25 - cut_off, q75 + cut_off
	# identify outliers
	outliers = [x for x in data if x < lower or x > upper]
	print('Identified outliers: %d' % len(outliers))
	print(outliers)
	# remove outliers
	outliers_removed = [x for x in data if x >= lower and x <= upper]
	print('Non-outlier observations: %d' % len(outliers_removed))
	return outliers_removed, outliers

def test_direct(outliers_removed, n_test, cfg):
	try:
		# never show warnings when grid searching, too noisy
		with catch_warnings():
			filterwarnings("ignore")
			error, expected, predicted = walk_forward_validation(outliers_removed, n_test, cfg)
			plot_sarima(expected, predicted)
			print(cfg, error)
	except:
		print("ERRO")

if __name__ == '__main__':
	# incendiosflorestais_focoscalor_brasil_1998-2017_copy
	series = read_csv('datasets/incendiosflorestais_focoscalor_brasil_1998-2017.csv', header=0, index_col=0)
	n_test = 60

	# preco_medio_mensal_revenda_gasolina_mg_2013_2020
	# series = read_csv('datasets/preco_medio_mensal_revenda_gasolina_mg_2013_2020.csv', header=0, index_col=0)
	# n_test = 24

	data = series.values
	print(data.shape)

	outliers_removed, outliers = remove_outliers(data)

	# model configs
	cfg_list = sarima_configs([12])


	# grid search
	scores = grid_search(outliers_removed, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:5]:
	  print(cfg, error)


	# Test the values direct
	# preco_medio_mensal_revenda_gasolina_mg_2013_2020
	# cfg = [(0, 1, 1), (1, 0, 2, 12), 'n']

	# # focos calor
	# cfg = [(2, 1, 1), (1, 0, 2, 12), 'n']

	# cfg = [(0, 0, 2), (2, 1, 0, 12), 'n']
	# cfg = [(0, 0, 2), (2, 1, 0, 12), 'c']
	
	# test_direct(outliers_removed, n_test, cfg)

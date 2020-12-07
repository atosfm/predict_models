# forecast monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from math import sqrt
from numpy import percentile
import datetime


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.3f, predicted=%.3f' % (testy, yhat))
	# estimate prediction error
	error = sqrt(mean_squared_error(test[:, -1], predictions))
	return error, test[:, -1], predictions, history

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


# load the dataset

series = read_csv('datasets/preco_medio_mensal_revenda_gasolina_mg_2013_2020.csv', header=0, index_col=0)
n_test = 24

# series = read_csv('datasets/incendiosflorestais_focoscalor_brasil_1998-2017_copy.csv', header=0, index_col=0)
# n_test = 60

values = series.values

# remove outliers
data, outliers = remove_outliers(values)

# transform the time series data into supervised learning
data = series_to_supervised(data, n_in=3)

# evaluate
rmsd, y, yhat, his = walk_forward_validation(data, n_test)
print('RMSD: %.3f' % rmsd)

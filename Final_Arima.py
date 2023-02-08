import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 
from sqlalchemy import create_engine

df = pd.read_csv(r"C:/Users/Tejas/OneDrive/Desktop/Final_Cement_Dataset.csv")
df.info()
df.columns
df.drop(columns={'Cement Production','Season','water_source','limestone','Coal'},inplace =True)

# Data Partition
Train = df.head(131)
Test = df.tail(12)


df1 = pd.read_excel('C:/Users/Tejas/OneDrive/Desktop/Test_Arima1.xlsx') 

###############prediction for Sales##################

tsa_plots.plot_acf(df.Sales, lags = 12)
tsa_plots.plot_pacf(df.Sales, lags = 12)


# ARIMA with AR = 4, MA = 6
model1 = ARIMA(Train.Sales, order = (12, 1, 6))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
start_index
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Sales, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Sales)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm
help(pm.auto_arima)

ar_model = pm.auto_arima(Train.Sales, start_p = 0, start_q = 0,
                      max_p = 16, max_q = 16, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)


# Best Parameters ARIMA
# ARIMA with AR=3, I = 1, MA = 5
model = ARIMA(Train.Sales, order = (4,1,1))
res = model.fit()
print(res.summary())


# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)


print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Sales, forecast_best))
print('Test RMSE: %.3f' % rmse_best)
# plot forecasts against actual outcomes
pyplot.plot(Test.Sales)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()


# checking both rmse of with and with out autoarima

print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)
# saving model whose rmse is low
# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
res1.save("model.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# Forecast for future 12 months
start_index = len(df)
end_index = start_index + 11
forecast = model.predict(start = start_index, end = end_index)

print(forecast)
pyplot.plot(Test.Sales)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()

############################Prediction for demand#########################
tsa_plots.plot_acf(df.Demand, lags = 12)
tsa_plots.plot_pacf(df.Demand, lags = 12)


# ARIMA with AR = 4, MA = 6
model1 = ARIMA(Train.Demand, order = (12, 1, 6))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
start_index
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Demand, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Demand)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm
help(pm.auto_arima)

ar_model = pm.auto_arima(Train.Demand, start_p = 0, start_q = 0,
                      max_p = 16, max_q = 16, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)


# Best Parameters ARIMA
# ARIMA with AR=3, I = 1, MA = 5
model = ARIMA(Train.Demand, order = (3,0,3))
res = model.fit()
print(res.summary())


# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)


print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Demand, forecast_best))
print('Test RMSE: %.3f' % rmse_best)
# plot forecasts against actual outcomes
pyplot.plot(Test.Demand)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()


# checking both rmse of with and with out autoarima

print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)
# saving model whose rmse is low
# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
res1.save("model1.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model1.pickle")

# Forecast for future 12 months
start_index = len(df)
end_index = start_index + 11
forecast = model.predict(start = start_index, end = end_index)

print(forecast)
pyplot.plot(Test.Demand)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()


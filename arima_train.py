import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def drop_obj_var_and_separate_endog_exog(df):
    df = df.select_dtypes(exclude=['object'])

    endog_var = 'precipitation'
    endog = df[endog_var]
    exog = df.drop(columns=[endog_var])

    return endog, exog

df_train = pd.read_csv('data/dexl/weather_station_1/weather_station_inmet_preprocessedtrain.csv')
df_test = pd.read_csv('data/dexl/weather_station_1/weather_station_inmet_preprocessedtest.csv')

endog_var = 'precipitation'
endog_train, exog_train = drop_obj_var_and_separate_endog_exog(df_train)
train_size = len(endog_train)

endog_test, exog_test = drop_obj_var_and_separate_endog_exog(df_test)

endog = pd.concat([endog_train, endog_test])

model = SARIMAX(endog_train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
model_fit = model.fit(disp=False)

forecast_horizon = len(endog_test)
forecast = model_fit.predict(start=train_size, end=train_size+forecast_horizon-1, exog=exog_test)

mae = mean_absolute_error(endog_test, forecast)
mse = mean_squared_error(endog_test, forecast)
print(f"""
        Mean Squared error (MSE): {mse}
      Mean Absolute error (MAE): {mae}
""")

plt.figure(figsize=(12, 6))
plt.plot(endog.index, endog, label='Data')
plt.plot(endog_test.index, forecast, label='Pred', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Prediction with SARIMAX')
plt.show()
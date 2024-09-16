# Precipitation Forecast

Code implementatation for Autoformer is based on the [github](https://github.com/thuml/Autoformer?tab=readme-ov-file)
Code based on the [github](https://github.com/cure-lab/LTSF-Linear)

Python version: 3.8

Data format should be:

```python
df_raw.columns: ['date', ...(other features), target feature]
```

To run ARIMA: `arima_train.py`.

To run Autoformer or DLinear: `run.py` passing model argument with name of the model you need.

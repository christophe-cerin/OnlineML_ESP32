# Time series

This page relates batch and Extreme-edge incremental learning for time series analysis. The latter is based on offline / batch learning method, but considering a 'sliding window' and not the full data.

## Batch learning and time series

You will find three types of methods: LSTM (Long short-term memory) and BI-LSTM (bidirectional long short-term memory), PROPHET, and TBATS (Trigonometric Box-Cox transform, ARMA errors, Trend, and Seasonal components). 

- Long short-term memory (LSTM)is a type of recurrent neural network (RNN) aimed at mitigating the vanishing gradient problem[2] commonly encountered by traditional RNNs.
- Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series with strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
- TBATS model is a powerful method for time series forecasting that can handle multiple seasonalities and complex patterns.



## Extreme-edge incremental learning and time series

# Time series

This page relates batch and Extreme-edge incremental learning for time series analysis. The latter is based on offline / batch learning method, but considering a 'sliding window' and not the full data.

We provide experimental results, based on random generated data or real data from an outside building (`TourPerretNoHeader.csv`). The humidity attribute of the dataset corresponds to the observed data for the building. We do forecasting for this attribute.

## Batch learning and time series

You will find three types of methods: LSTM (Long short-term memory) and BI-LSTM (bidirectional long short-term memory), PROPHET, and TBATS (Trigonometric Box-Cox transform, ARMA errors, Trend, and Seasonal components). 

- Long short-term memory (LSTM)is a type of recurrent neural network (RNN) aimed at mitigating the vanishing gradient problem[2] commonly encountered by traditional RNNs.
- Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series with strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
- TBATS model is a powerful method for time series forecasting that can handle multiple seasonalities and complex patterns.

Figure 1 illustrates the humidity values over 360 time intervals, represented by the plot in red. The green and blue lines indicate the training data, while the grey line represents the predictions. The Python code adheres to traditional methodologies. Once the dataset is prepared, it is imperative to partition it into training and testing subsets to assess the model's performance on data unseen during training. Subsequently, a Long Short-Term Memory (LSTM) model is employed. Upon fitting the model with the training data, predictions are generated on the test set. This approach provides insights into the model's efficacy in forecasting humidity based on novel input data.

<figure>
  <img src="Images/LSTM.png" alt="My image caption">
  <figcaption><b>Fig. 1:</b> Exploring Tour Perret dataset with LSTM</figcaption>
</figure>

Figure 2 relates to a preriodical signal perturbated with 'noise'

<figure>
  <img src="Images/PROPHET.png" alt="My image caption">
  <figcaption><b>Fig. 2:</b> Exploring a random generated dataset with PROPHET</figcaption>
</figure>


## Extreme-edge incremental learning and time series

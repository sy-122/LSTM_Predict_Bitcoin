# Bitcoin price prediction with LSTM Model

This is a simple LSTM model that predicts the price of Bitcoin. 
Long Short-Term Memory (LSTM) is a specialized type of recurrent neural network (RNN) engineered to process and make predictions on sequential data, such as time series information. Unlike conventional feedforward neural networks, which analyze each input in isolation, LSTMs are equipped with loops that enable the retention of relevant information over time, thereby fostering a deeper understanding of temporal patterns. One of its greatest advantages lies in its ability to establish connections between past observations and the current task at hand.

The fundamental components of an LSTM unit encompass:
- **Cell State**: This element functions as the memory reservoir of the network, enabling it to selectively retain or discard information across successive time steps.
- **Input Gate**: Dictates the extent to which new data is incorporated into the cell state.
- **Forget Gate**: Governs the extent to which obsolete information is purged from the cell state.
- **Output Gate**: Determines the degree to which the cell state is utilized in generating the output.

For a comprehensive elucidation of LSTM networks, I recommend perusing the following resource: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah, which offers an insightful exploration of LSTM architecture and functionality.

The code responsible for conducting predictions can be found in `model.py`, with its implementation drawing inspiration from the tutorial available at: [YouTube Tutorial on LSTM Time Series Prediction](https://www.youtube.com/watch?v=GFSiL6zEZF0).

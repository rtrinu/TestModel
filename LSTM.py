import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def LSTM_model_creation():
    df = pd.read_csv('Compound_AI.csv')
    df = df.dropna()
    features = ['Close', 'Close_Shifted', 'RSI', 'SMA_50', 'EMA_20', 'Compound', 'Open', 'Open_Shifted']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    sequence_length = 60
    x =[]
    y=[]

    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i,0])

    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from statsmodels.tsa.ar_model import AutoReg


class SARHMM():
    def __init__(self, filename: str):
        self.features = None
        self.returns = None
        self.data = None
        self.filename = filename

    def load_and_preprocess_data(self):
        self.data = pd.read_csv(self.filename)
        self.data['Close'] = pd.to_numeric(self.data['Close'], errors='coerce')
        self.data['Previous_Close'] = pd.to_numeric(self.data['Previous_Close'], errors='coerce')
        self.data['RSI'] = pd.to_numeric(self.data['RSI'], errors='coerce')
        self.data['SMA_50'] = pd.to_numeric(self.data['SMA_50'], errors='coerce')
        self.data['EMA_20'] = pd.to_numeric(self.data['EMA_20'], errors='coerce')
        self.data['MACD'] = pd.to_numeric(self.data['MACD'], errors='coerce')
        self.data['MACD_signal'] = pd.to_numeric(self.data['MACD_signal'], errors='coerce')
        self.data['MACD_histogram'] = pd.to_numeric(self.data['MACD_histogram'], errors='coerce')
        self.data['Signal'] = pd.to_numeric(self.data['Signal'], errors='coerce')
        self.data['Compound Sentiment'] = pd.to_numeric(self.data['Compound Sentiment'], errors='coerce')

        self.data.dropna(inplace=True)
        self.data['Returns'] = np.log(self.data['Close'] / self.data['Previous_Close'])

        self.features = self.data[
            ['RSI', 'SMA_50', 'EMA_20', 'MACD', 'MACD_signal', 'MACD_histogram', 'Signal', 'Compound Sentiment']].values
        self.returns = self.data['Returns'].values

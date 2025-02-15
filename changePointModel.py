import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
class ChangePointModel():
    def __init__(self, filename:str):
        self.filename = filename
        self.data = None
        self.initalise()

    def load_and_process_dataset(self):
        self.data = pd.read_csv(self.filename)

    def calculate_returns(self):
        self.data['Close'] = pd.to_numeric(self.data['Close'],errors='coerce')
        self.data['Returns'] = self.data['Close'].pct_change()

    def calculate_volatility(self, window=20):
        self.data['Volatility'] = self.data['Returns'].rolling(window).std()

    def detect_change_point(self, column='Returns',model='l2',penalty=10):
        signal = self.data[column].dropna().values
        algo = rpt.Pelt(model=model).fit(signal)
        change_points = algo.predict(pen=penalty)
        rpt.display(signal,change_points)
        plt.show()



    def initalise(self):
        self.load_and_process_dataset()
        self.calculate_returns()
        self.calculate_volatility()
        self.detect_change_point()
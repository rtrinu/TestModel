import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class ChangePointModel():
    def __init__(self, filename: str):
        self.filename = filename
        self.data = None
        self.initalise()

    def load_and_process_dataset(self):
        self.data = pd.read_csv(self.filename)
        if 'Close' not in self.data.columns:
            raise ValueError("The dataset must contain a 'Close' column.")

    def calculate_returns(self):
        self.data['Close'] = pd.to_numeric(self.data['Close'], errors='coerce')
        self.data['Returns'] = self.data['Close'].pct_change()

    def calculate_volatility(self, window=20):
        self.data['Volatility'] = self.data['Returns'].rolling(window).std()

    def evaluate_change_points(self, column='Returns', model='rbf', penalty=0.6): #adjust penalty
        """Detect change points and visualize them for qualitative evaluation."""
        signal = self.data[column].dropna().values
        algo = rpt.Pelt(model=model).fit(signal)
        change_points = algo.predict(pen=penalty)

        self.data['Date'] = self.data['Date'].astype(str)

        plt.figure(figsize=(10, 6))
        plt.plot(self.data['Date'], self.data['Close'], label="Stock Price")
        plt.scatter(self.data['Date'].iloc[change_points], self.data['Close'].iloc[change_points], color='red',
                    label="Detected Change Points")
        plt.title(f"Change Point Detection for {column} using {model}")
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def initalise(self):
        self.load_and_process_dataset()
        self.calculate_returns()
        self.calculate_volatility()
        # self.detect_change_point()
        self.evaluate_change_points()

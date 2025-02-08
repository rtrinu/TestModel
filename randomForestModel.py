import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class randomForestModel():
    def __init__(self, filename: str):
        self.filename = filename
        self.feature_columns = [
            'Previous_Close', 'RSI', 'SMA_50', 'EMA_20', 'MACD', 'MACD_signal', 'MACD_histogram', 'Signal',
            'Compound Sentiment'
        ]
        self.target_column = 'Close'
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.initalise()

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.filename)
        df.dropna()

        x = df[self.feature_columns]
        y = df[self.target_column].shift(-1).dropna()
        x = x.iloc[:-1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error on test data: {mse}")
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        print("\nActual vs Predicted Values:")
        print(comparison_df.head())

    def predict(self, x_input):
        return self.model.predict(x_input)

    def initalise(self):
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_and_preprocess_data()
        self.train_model(self.x_train, self.y_train)
        self.evaluate_model(self.x_test, self.y_test)
        future_data = self.x_test.head(5)
        predictions = self.predict(future_data)
        print(f"Predictions: {predictions}")

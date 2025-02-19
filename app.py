import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
import random as rnd
import os
from flask import Flask, render_template, send_file

# Initialize Flask app
app = Flask(__name__)

def plot_close_data():
    companies = ["AAPL", "NVDA", "MSFT", "AMZN","META","GOOGL","AVGO"]

    # Define the time period (last 5 days)
    end_date = dt.today()
    start_date = end_date - timedelta(days=5)

    # Randomly select two companies
    first_random_int = rnd.randint(0, 6)
    first_company = companies[first_random_int]

    del companies[first_random_int]  # Remove the selected company

    second_random_int = rnd.randint(0, 5)
    second_company = companies[second_random_int]

    # Download stock data
    first_company_data = yf.download(first_company, start=start_date, end=end_date)
    second_company_data = yf.download(second_company, start=start_date, end=end_date)

    # Extract closing prices
    fc_prices = first_company_data['Close']
    sc_prices = second_company_data['Close']

    first_date = first_company_data['Date'].iloc[0]
    last_date = first_company_data['Date'].iloc[-1]


    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.ylim(100, 1000)
    plt.xlim(first_date, last_date)

    plt.plot(fc_prices, label=f'{first_company}', color='blue')
    plt.plot(sc_prices, label=f'{second_company}', color='green')

    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Closing Prices of Two Random Companies')
    
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the image in-memory (instead of saving to static directory)
    image_path = "static/plot.png"
    plt.savefig(image_path)
    plt.close()  # Close the figure to free memory

    return image_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plot')
def serve_plot():
    """This route generates a new plot on each refresh"""
    plot_close_data()  # Generate a fresh plot
    return send_file("static/plot.png", mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)

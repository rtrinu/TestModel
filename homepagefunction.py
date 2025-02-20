import yfinance as yf
import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt, timedelta
import pandas as pd

def plot_close_data(image_filename):
    companies = ["AAPL", "NVDA", "MSFT", "AMZN","META","GOOGL","AVGO"]

    # Define the time period (last 5 days)
    end_date = dt.today()
    start_date = end_date - timedelta(days=90)

    # Randomly select two companies
    first_random_int = rnd.randint(0, 6)
    first_company = companies[first_random_int]

    del companies[first_random_int]  # Remove the selected company

    second_random_int = rnd.randint(0, 5)
    second_company = companies[second_random_int]

    # Download stock data
    first_company_data = yf.download(first_company, start=start_date, end=end_date)
    second_company_data = yf.download(second_company, start=start_date, end=end_date)
    first_company_data = first_company_data.reset_index()
    second_company_data = second_company_data.reset_index()
    
    # Extract closing prices
    fc_prices = first_company_data['Close']
    sc_prices = second_company_data['Close']

    dates = pd.to_datetime(first_company_data['Date'])
    first_date = dates.min()
    last_date = dates.max()

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.xlim(first_date, last_date)

    plt.plot(dates, fc_prices, label=f'{first_company}', color='blue')
    plt.plot(dates, sc_prices, label=f'{second_company}', color='green')

    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Closing Prices of Two Random Companies')
    
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the image with the provided filename (this will overwrite the previous plot.png)
    image_path = f"static/{image_filename}"
    plt.savefig(image_path)
    plt.close()  # Close the figure to free memory

    return image_filename

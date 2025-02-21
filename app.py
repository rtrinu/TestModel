import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
import random as rnd
import os
from flask import Flask, render_template, send_file, request
from homepagefunction import plot_close_data
from stock import Stock

app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    img_filename = 'plot.png'
    plot_image_path = plot_close_data(img_filename)
    return render_template('index.html', image_path = plot_image_path)


@app.route('/stock_input')
def stock_input():
    return render_template('stockInput.html')

@app.route('/static/<filename>')
def send_image(filename):
    return send_file(f'static/{filename}')

@app.route('/stock',methods=['GET'])
def get_stock_data():
    stock_symbol = request.args.get('stock','').upper()
    if not stock_symbol:
        return "Input a valid symbol"

    user_stock = Stock(stock_symbol)
    stock_data = user_stock.display_information()
    return render_template('stockDisplay.html', stock_data = stock_data)

if __name__ == '__main__':
    app.run()

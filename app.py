from flask import Flask, render_template, request
from stock import Stock
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

"""
@app.route('/predict',methods=['POST'])
def predict():
    stock_symbol = request.form.get('stock_symbol')

    stock = Stock(stock_symbol)
""" 

if __name__ == '__main__':
    app.run(debug=True)
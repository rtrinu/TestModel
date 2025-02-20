import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
import random as rnd
import os
from flask import Flask, render_template, send_file
from homepagefunction import plot_close_data


# Initialize Flask app
app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')



@app.route('/')
def index():
    img_filename = 'plot.png'
    plot_image_path = plot_close_data(img_filename)
    return render_template('index.html', image_path = plot_image_path)

@app.route('/static/<filename>')
def send_image(filename):
    return send_file(f'static/{filename}')



if __name__ == '__main__':
    app.run(debug=True)

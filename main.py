from stock import Stock
import datetime as dt
import os


def main():
    stock = input("Input a Stock or Stock Symbol: ")
    
    example = Stock(stock)
    example.train_ai_models()
    

if __name__ == "__main__":
    main()

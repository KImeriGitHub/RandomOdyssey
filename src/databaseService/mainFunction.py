import yaml # the library is pyyaml not yaml
import yfinance as yf
import pandas as pd
import datetime as dt
import os

# Step 1: Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Step 2: Download stock information
def download_stock_data(stocks):
    stock_data = {}
    for ticker in [stocks[0], stocks[1]]:
        stock_info = yf.Ticker(ticker)
        print(stock_info.financials)
        stock_data[ticker] = yf.download(ticker, dt.date.today()-dt.timedelta(36500),dt.date.today(),interval='1d')
    return stock_data

# Main function
def mainFunction():
    yaml_file = 'src/databaseService/stockTickers.yaml'
    file_path = os.path.join(os.getcwd(), yaml_file)
    data = load_yaml(file_path)
    
    stock_exchange = data[0]['stockExchange']
    stock_list = data[0]['stocks']
    print(stock_exchange)
    if stock_list:
        stock_data = download_stock_data(stock_list)
        for ticker, info in stock_data.items():
            print(f"Stock data for {ticker}:")
            print(info)
    else:
        print("No stocks found in the YAML file.")
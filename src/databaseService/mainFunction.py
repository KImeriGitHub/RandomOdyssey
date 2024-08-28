import pyyaml
import yfinance as yf

# Step 1: Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = pyyaml.safe_load(file)
    return data

# Step 2: Download stock information
def download_stock_data(stocks):
    stock_data = {}
    for ticker in stocks:
        stock_info = yf.Ticker(ticker)
        stock_data[ticker] = stock_info.history(period="5d")  # Download last 5 days of data
    return stock_data

# Main function
def mainFunction():
    print("Hello, World!")
    yaml_file = 'stockTickers.yaml'
    data = load_yaml(yaml_file)
    
    stock_exchange = data.get('stockExchange')
    stock_list = data.get('stocks', [])
    
    if stock_list:
        stock_data = download_stock_data(stock_list)
        for ticker, info in stock_data.items():
            print(f"Stock data for {ticker}:")
            print(info)
    else:
        print("No stocks found in the YAML file.")
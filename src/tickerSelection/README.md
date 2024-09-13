# Gather Tickers Manually

## Structure
The yaml file of tickers, stockTickers.yaml, has to be of the following form:
- stockExchange: NYSE
  stocks:
    - MMM
    - DDD
    - AOS
    - .......
- stockExchange: NASDAQ
  stocks:
    - ADBE
    - ABNB
    - GOOGL
    - .......
- stockExchange: SIX
  stocks:
    - MMM
    - ABBN
    - ABT
    - .......
.......

## Ticker Symbols
Symbols may be of the form MMM or MMM.sw. Upper or Lower case.
Any ID will be tried as well but is not guaranteed to succeed.

## Code Flow
The class LoadStocks.py in src/databaseService takes on the stockTickers.yaml, process every ticker and 
established a list of all working stocks in src/stockGroups with name group_all.ymal
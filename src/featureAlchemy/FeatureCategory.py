import numpy as np
import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetDataPolars import AssetDataPolars

class FeatureCategory():
    operator = "alphavantage"
    
    DEFAULT_PARAMS = {
        'timesteps': 10,
    }
    
    # Class-level default parameters
    cat_alphavantage = [
        'other', 
        'industrials',
        'healthcare', 
        'technology', 
        'financial-services', 
        'real-estate', 
        'energy', 
        'consumer-cyclical', 
    ]
    
    cat_yfinance =[
        'other', 'industrials', 'healthcare', 'technology', 'utilities', 
        'financial-services', 'basic-materials', 'real-estate', 
        'consumer-defensive', 'energy', 'communication-services', 
        'consumer-cyclical'
    ]
    
    def __init__(self, asset: AssetDataPolars, params: dict = None):
            
        self.asset = asset
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.timesteps = self.params['timesteps']
        
        self.cat = self.cat_alphavantage if self.operator == "alphavantage" else self.cat_yfinance
    
    def getFeatureNames(self) -> list[str]:
        features_names = ["Category_" + val for val in self.cat]
        features_names += ["Category_inSnP500", "Category_inNas100"]
            
        return features_names
    
    def getTimeFeatureNames(self) -> list[str]:
        return []
    
    def apply(self, scaleToNiveau: float):
        sector = self.asset.sector
        
        # Create a one-hot encoding where the category matches the sector
        features = np.zeros(len(self.cat)+2, dtype=np.float32)
        features[:len(self.cat)] = np.array([1.0 if category == sector else 0.0 for category in self.cat])
        features[len(self.cat)] = self.asset.ticker in self.snp500tickers
        features[len(self.cat)+1] = self.asset.ticker in self.nas100tickers
        
        return features*scaleToNiveau
    
    def apply_timeseries(self, date: pd.Timestamp, idx: int = None) -> np.ndarray:
        return np.empty((self.timesteps, 0), dtype=np.float32)
    
    
    nas100tickers = [
        "BF-B",
        "BRK-B",
        "ADBE",
        "ABNB",
        "GOOGL",
        "GOOG",
        "AMZN",
        "AMD",
        "AEP",
        "AMGN",
        "ADI",
        "ANSS",
        "AAPL",
        "AMAT",
        "ARM",
        "ASML",
        "AZN",
        "TEAM",
        "ADSK",
        "ADP",
        "BKR",
        "BIIB",
        "BKNG",
        "AVGO",
        "CDNS",
        "CDW",
        "CHTR",
        "CTAS",
        "CSCO",
        "CCEP",
        "CTSH",
        "CMCSA",
        "CEG",
        "CPRT",
        "CSGP",
        "COST",
        "CRWD",
        "CSX",
        "DDOG",
        "DXCM",
        "FANG",
        "DLTR",
        "DASH",
        "EA",
        "EXC",
        "FAST",
        "FTNT",
        "GEHC",
        "GILD",
        "GFS",
        "HON",
        "IDXX",
        "ILMN",
        "INTC",
        "INTU",
        "ISRG",
        "KDP",
        "KLAC",
        "KHC",
        "LRCX",
        "LIN",
        "LULU",
        "MAR",
        "MRVL",
        "MELI",
        "META",
        "MCHP",
        "MU",
        "MSFT",
        "MRNA",
        "MDLZ",
        "MDB",
        "MNST",
        "NFLX",
        "NVDA",
        "NXPI",
        "ORLY",
        "ODFL",
        "ON",
        "PCAR",
        "PANW",
        "PAYX",
        "PYPL",
        "PDD",
        "PEP",
        "QCOM",
        "REGN",
        "ROP",
        "ROST",
        "SBUX",
        "SMCI",
        "SNPS",
        "TTWO",
        "TMUS",
        "TSLA",
        "TXN",
        "TTD",
        "VRSK",
        "VRTX",
        "WBD",
        "WDAY",
        "XEL",
        "ZS"
        ]
    
    snp500tickers = [
        "MMM",
        "AOS",
        "ABT",
        "ABBV",
        "ACN",
        "ADBE",
        "AMD",
        "AES",
        "AFL",
        "A",
        "APD",
        "ABNB",
        "AKAM",
        "ALB",
        "ARE",
        "ALGN",
        "ALLE",
        "LNT",
        "ALL",
        "GOOGL",
        "GOOG",
        "MO",
        "AMZN",
        "AMCR",
        "AMTM",
        "AEE",
        "AEP",
        "AXP",
        "AIG",
        "AMT",
        "AWK",
        "AMP",
        "AME",
        "AMGN",
        "APH",
        "ADI",
        "ANSS",
        "AON",
        "APA",
        "AAPL",
        "AMAT",
        "APTV",
        "ACGL",
        "ADM",
        "ANET",
        "AJG",
        "AIZ",
        "T",
        "ATO",
        "ADSK",
        "ADP",
        "AZO",
        "AVB",
        "AVY",
        "AXON",
        "BKR",
        "BALL",
        "BAC",
        "BAX",
        "BDX",
        "BRK-B",
        "BBY",
        "TECH",
        "BIIB",
        "BLK",
        "BX",
        "BK",
        "BA",
        "BKNG",
        "BWA",
        "BSX",
        "BMY",
        "AVGO",
        "BR",
        "BRO",
        "BF-B",
        "BLDR",
        "BG",
        "BXP",
        "CHRW",
        "CDNS",
        "CZR",
        "CPT",
        "CPB",
        "COF",
        "CAH",
        "KMX",
        "CCL",
        "CARR",
        "CTLT",
        "CAT",
        "CBRE",
        "CDW",
        "CE",
        "COR",
        "CNC",
        "CNP",
        "CF",
        "CRL",
        "SCHW",
        "CHTR",
        "CVX",
        "CMG",
        "CB",
        "CHD",
        "CI",
        "CINF",
        "CTAS",
        "CSCO",
        "C",
        "CFG",
        "CLX",
        "CME",
        "CMS",
        "KO",
        "CTSH",
        "CL",
        "CMCSA",
        "CAG",
        "COP",
        "ED",
        "STZ",
        "CEG",
        "COO",
        "CPRT",
        "GLW",
        "CPAY",
        "CTVA",
        "CSGP",
        "COST",
        "CTRA",
        "CRWD",
        "CCI",
        "CSX",
        "CMI",
        "CVS",
        "DHR",
        "DRI",
        "DVA",
        "DAY",
        "DECK",
        "DE",
        "DELL",
        "DAL",
        "DVN",
        "DXCM",
        "FANG",
        "DLR",
        "DFS",
        "DG",
        "DLTR",
        "D",
        "DPZ",
        "DOV",
        "DOW",
        "DHI",
        "DTE",
        "DUK",
        "DD",
        "EMN",
        "ETN",
        "EBAY",
        "ECL",
        "EIX",
        "EW",
        "EA",
        "ELV",
        "EMR",
        "ENPH",
        "ETR",
        "EOG",
        "EPAM",
        "EQT",
        "EFX",
        "EQIX",
        "EQR",
        "ERIE",
        "ESS",
        "EL",
        "EG",
        "EVRG",
        "ES",
        "EXC",
        "EXPE",
        "EXPD",
        "EXR",
        "XOM",
        "FFIV",
        "FDS",
        "FICO",
        "FAST",
        "FRT",
        "FDX",
        "FIS",
        "FITB",
        "FSLR",
        "FE",
        "FI",
        "FMC",
        "F",
        "FTNT",
        "FTV",
        "FOXA",
        "FOX",
        "BEN",
        "FCX",
        "GRMN",
        "IT",
        "GE",
        "GEHC",
        "GEV",
        "GEN",
        "GNRC",
        "GD",
        "GIS",
        "GM",
        "GPC",
        "GILD",
        "GPN",
        "GL",
        "GDDY",
        "GS",
        "HAL",
        "HIG",
        "HAS",
        "HCA",
        "DOC",
        "HSIC",
        "HSY",
        "HES",
        "HPE",
        "HLT",
        "HOLX",
        "HD",
        "HON",
        "HRL",
        "HST",
        "HWM",
        "HPQ",
        "HUBB",
        "HUM",
        "HBAN",
        "HII",
        "IBM",
        "IEX",
        "IDXX",
        "ITW",
        "INCY",
        "IR",
        "PODD",
        "INTC",
        "ICE",
        "IFF",
        "IP",
        "IPG",
        "INTU",
        "ISRG",
        "IVZ",
        "INVH",
        "IQV",
        "IRM",
        "JBHT",
        "JBL",
        "JKHY",
        "J",
        "JNJ",
        "JCI",
        "JPM",
        "JNPR",
        "K",
        "KVUE",
        "KDP",
        "KEY",
        "KEYS",
        "KMB",
        "KIM",
        "KMI",
        "KKR",
        "KLAC",
        "KHC",
        "KR",
        "LHX",
        "LH",
        "LRCX",
        "LW",
        "LVS",
        "LDOS",
        "LEN",
        "LLY",
        "LIN",
        "LYV",
        "LKQ",
        "LMT",
        "L",
        "LOW",
        "LULU",
        "LYB",
        "MTB",
        "MRO",
        "MPC",
        "MKTX",
        "MAR",
        "MMC",
        "MLM",
        "MAS",
        "MA",
        "MTCH",
        "MKC",
        "MCD",
        "MCK",
        "MDT",
        "MRK",
        "META",
        "MET",
        "MTD",
        "MGM",
        "MCHP",
        "MU",
        "MSFT",
        "MAA",
        "MRNA",
        "MHK",
        "MOH",
        "TAP",
        "MDLZ",
        "MPWR",
        "MNST",
        "MCO",
        "MS",
        "MOS",
        "MSI",
        "MSCI",
        "NDAQ",
        "NTAP",
        "NFLX",
        "NEM",
        "NWSA",
        "NWS",
        "NEE",
        "NKE",
        "NI",
        "NDSN",
        "NSC",
        "NTRS",
        "NOC",
        "NCLH",
        "NRG",
        "NUE",
        "NVDA",
        "NVR",
        "NXPI",
        "ORLY",
        "OXY",
        "ODFL",
        "OMC",
        "ON",
        "OKE",
        "ORCL",
        "OTIS",
        "PCAR",
        "PKG",
        "PLTR",
        "PANW",
        "PARA",
        "PH",
        "PAYX",
        "PAYC",
        "PYPL",
        "PNR",
        "PEP",
        "PFE",
        "PCG",
        "PM",
        "PSX",
        "PNW",
        "PNC",
        "POOL",
        "PPG",
        "PPL",
        "PFG",
        "PG",
        "PGR",
        "PLD",
        "PRU",
        "PEG",
        "PTC",
        "PSA",
        "PHM",
        "QRVO",
        "PWR",
        "QCOM",
        "DGX",
        "RL",
        "RJF",
        "RTX",
        "O",
        "REG",
        "REGN",
        "RF",
        "RSG",
        "RMD",
        "RVTY",
        "ROK",
        "ROL",
        "ROP",
        "ROST",
        "RCL",
        "SPGI",
        "CRM",
        "SBAC",
        "SLB",
        "STX",
        "SRE",
        "NOW",
        "SHW",
        "SPG",
        "SWKS",
        "SJM",
        "SW",
        "SNA",
        "SOLV",
        "SO",
        "LUV",
        "SWK",
        "SBUX",
        "STT",
        "STLD",
        "STE",
        "SYK",
        "SMCI",
        "SYF",
        "SNPS",
        "SYY",
        "TMUS",
        "TROW",
        "TTWO",
        "TPR",
        "TRGP",
        "TGT",
        "TEL",
        "TDY",
        "TFX",
        "TER",
        "TSLA",
        "TXN",
        "TXT",
        "TMO",
        "TJX",
        "TSCO",
        "TT",
        "TDG",
        "TRV",
        "TRMB",
        "TFC",
        "TYL",
        "TSN",
        "USB",
        "UBER",
        "UDR",
        "ULTA",
        "UNP",
        "UAL",
        "UPS",
        "URI",
        "UNH",
        "UHS",
        "VLO",
        "VTR",
        "VLTO",
        "VRSN",
        "VRSK",
        "VZ",
        "VRTX",
        "VTRS",
        "VICI",
        "V",
        "VST",
        "VMC",
        "WRB",
        "GWW",
        "WAB",
        "WBA",
        "WMT",
        "DIS",
        "WBD",
        "WM",
        "WAT",
        "WEC",
        "WFC",
        "WELL",
        "WST",
        "WDC",
        "WY",
        "WMB",
        "WTW",
        "WYNN",
        "XEL",
        "XYL",
        "YUM",
        "ZBRA",
        "ZBH",
        "ZTS"]
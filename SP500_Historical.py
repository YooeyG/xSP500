
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import yfinance as yf
import types
from concurrent.futures import ThreadPoolExecutor, as_completed

import concurrent.futures
import time



import matplotlib.pyplot as plt

# URL of the Wikipedia page
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Fetch the HTML content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract the 'constituents' table
table = soup.find('table', {'id': 'constituents'})
current_constituents = pd.read_html(str(table))[0]


# Extract the 'changes' table
table = soup.find('table', {'id': 'changes'})
spx_changes = pd.read_html(str(table), header=0)[0]
spx_changes = spx_changes.drop(spx_changes.index[0:2])


spx_changes.columns = ['Date', 'AddTicker', 'AddName', 'RemovedTicker', 'RemovedName', 'Reason']
spx_changes['Date'] = pd.to_datetime(spx_changes['Date'], format='%B %d, %Y')
spx_changes['year'] = spx_changes['Date'].dt.year
spx_changes['month'] = spx_changes['Date'].dt.month

print(spx_changes)



# Initialize the current month and create a sequence of months
current_month = datetime.today().replace(day=1)
month_sequence = pd.date_range(start='1990-01-01', end=current_month, freq='MS')[::-1]



# Prepare the current constituents list with an added date
spx_stocks = current_constituents.assign(Date=current_month).rename(columns={'Symbol': 'Ticker', 'Security': 'Name'})
last_run_stocks = spx_stocks.copy()

# Iterate through months, working backwards
for d in month_sequence[1:]:
    y, m = d.year, d.month

    # Filter changes for the specific year and month
    changes = spx_changes[(spx_changes['year'] == y) & (spx_changes['month'] == m)]

    # Remove added tickers
    tickers_to_keep = last_run_stocks[~last_run_stocks['Ticker'].isin(changes['AddTicker'])].assign(Date=d)

    # Add back the removed tickers
    tickers_to_add = changes[changes['RemovedTicker'] != '']
    tickers_to_add = tickers_to_add.assign(Date=d, Ticker=changes['RemovedTicker'], Name=changes['RemovedName'])

    # Combine tickers to keep and tickers to add
    this_month = pd.concat([tickers_to_keep, tickers_to_add], ignore_index=True)
    spx_stocks = pd.concat([spx_stocks, this_month], ignore_index=True)

    last_run_stocks = this_month

spx_stocks


#DATA GARABAGE CLEAN UP - DRAFT 2
def clean_namespace_except(keep_list):
    global_vars = globals().copy()
    for name, val in global_vars.items():
        # Check if the object is a module or a function, and if so, skip it
        if isinstance(val, (types.ModuleType, types.FunctionType)):
            continue
        # If the variable is not in the keep list, delete it from the global scope
        if name not in keep_list:
            del globals()[name]

clean_namespace_except(['spx_stocks'])


############################# YFinance PUll ########################################
distinct_tickers = spx_stocks['Ticker'].unique()


# Group by 'Date' and count the constituents
date_counts = spx_stocks.groupby('Date').size().reset_index(name='count')
start_date = "2000-01-01"
end_date = "2023-12-20"


#THIS NEEDS TO BE TREADING TO GO FASTER - THREADING HAS RATE LIMITS
def download_stock_data(tickers, start_date, end_date):
    stock_data_list = []
    error_tickers = []

    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if stock_data.empty:
                raise ValueError(f"No data found for {ticker}")

            # Reindex the stock_data with a date range and use 'ffill' to forward-fill missing values
            date_range = pd.date_range(start=start_date, end=end_date)
            stock_data = stock_data.reindex(date_range).fillna(method='ffill')

            # If the first row has NaN after reindexing, fill with zeros
            stock_data.fillna(0, inplace=True)
            
            stock_data['Ticker'] = ticker
            
            # Reset the index to add the date as a column
            stock_data.reset_index(inplace=True)
            stock_data.rename(columns={'index': 'Date'}, inplace=True)
            
            stock_data_list.append(stock_data)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            error_tickers.append(ticker)

    all_stocks_data = pd.concat(stock_data_list) if stock_data_list else pd.DataFrame()
    all_stocks_data.reset_index(inplace=True, drop=True)

    return all_stocks_data, error_tickers

individual_stocks_data, failed_tickers = download_stock_data(distinct_tickers, start_date, end_date)


############################# BEGIN SIMFIN ########################################

#OPTIMIZED CODE
def sim_stock_data(tickers):
    base_url = "https://backend.simfin.com/api/v3/companies/prices/compact?ticker="
    headers = {
        "accept": "application/json",
        "Authorization": "553bc9c1-309d-4b3c-9240-44d0376d9a41"
    }

    all_data = []
    request_counter = 0
    start_time = time.time()

    for ticker in tickers:
        if not isinstance(ticker, str):  # Skip non-string tickers
            print(f"Skipping non-string ticker: {ticker}")
            continue

        # Rate limiting
        if request_counter >= 2:
            elapsed_time = time.time() - start_time
            if elapsed_time < 1:
                time.sleep(1 - elapsed_time)
            start_time = time.time()
            request_counter = 0

        url = base_url + ticker
        response = requests.get(url, headers=headers)
        request_counter += 1

        if response.status_code == 200:
            print(ticker)
            data = response.json()
            all_data.extend(data)
        else:
            print(f"Failed to download data for {ticker}. Status Code: {response.status_code}")

    return all_data



#CREATE DATAFRAME
stock_data = sim_stock_data(failed_tickers)

all_data_rows = []

for stock_dict in stock_data:
    data = stock_dict['data']
    stock = stock_dict['ticker']
    id_ = stock_dict['id']
    isin = stock_dict['isin']
    name_ = stock_dict['name']
    
    
    for d in data:
        row = d + [stock] + [id_] + [isin] + [name_]  # Assuming 'd' is a list of values
        all_data_rows.append(row)

column_names = ['Date',
                'Dividend Paid',
                'Common Shares Outstanding',
                'Last Closing Price',
                'Adj Close',
                'High',
                'Low',
                'Open',
                'Volume',
                'Ticker',
                'id',
                'isin',
                'name']  

sim_stocks_df = pd.DataFrame(all_data_rows, columns=column_names)
sim_stocks_df ['Source'] = 'Sim'
individual_stocks_data['Source'] = 'Sim'



clean_namespace_except(['all_data_rows', 
                        'sim_stocks_df', 
                        'distinct_tickers', 
                        'start_date', 
                        'end_date', 
                        'failed_tickers', 
                        'individual_stocks_data', 
                        'spx_stocks'])


############################# BEGIN UNION DATA SETS ########################################

combined_df = pd.concat([individual_stocks_data, sim_stocks_df], ignore_index=True, sort=False)

combined_distinct_tickers = combined_df['Ticker'].unique()
print(combined_df.head(10))





result = combined_df.groupby('Ticker')['Date'].agg(['min', 'max'])
print(result)



dedupped_df = spx_stocks[['Ticker', 'Name']].drop_duplicates()
dedupped_df.rename(columns={'Name': 'Company'}, inplace=True)
result  = pd.merge(result, dedupped_df, on='Ticker', how='left')








############################# END UNION SETS ########################################





SPY_list = ['SPY']
individual_stocks_data, failed_tickers = download_stock_data(SPY_list , start_date, end_date)


def compare_dates(df1, df2):
    dates_df1 = set(df1['Date'])
    dates_df2 = set(df2['Date'])

    unique_dates_df1 = dates_df1 - dates_df2
    unique_dates_df2 = dates_df2 - dates_df1

    return unique_dates_df1, unique_dates_df2































# Compare dates between the two datasets
unique_dates_individual, unique_dates_spy = compare_dates(individual_stocks_data, spy_data)
print("Unique dates in individual stocks dataset:", unique_dates_individual)
print("Unique dates in SPY dataset:", unique_dates_spy)


def drop_unique_dates(df1, df2):
    dates_df1 = set(df1['Date'])
    dates_df2 = set(df2['Date'])

    unique_dates_df1 = dates_df1 - dates_df2

    return df1[~df1['Date'].isin(unique_dates_df1)]


# Drop any dates in individual_stocks_data that are unique when compared to spy_data
individual_stocks_data_updated = drop_unique_dates(individual_stocks_data, spy_data)





















prices_df = prices[(prices['date'] >= '1990-01-01') & (prices['ticker'].isin(distinct_tickers))]

spx_stocks['month'] = spx_stocks['Date'].dt.month
spx_stocks['year'] = spx_stocks['Date'].dt.year
spx_stocks['ticker'] = spx_stocks['Ticker']

prices_df['month'] = prices_df['date'].dt.month
prices_df['year'] = prices_df['date'].dt.year

merged_df = pd.merge(prices_df, spx_stocks, on=['month', 'year', 'ticker'], how='left')
merged_df['inSPX'] = ~merged_df['Ticker'].isna()
final_df = merged_df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'closeunadj', 'inSPX']]
final_df













# Group by 'Date' and count the constituents
date_counts = spx_stocks.groupby('Date').size().reset_index(name='count')
start_date = "2023-12-01"
end_date = "2023-12-19"


date_range = pd.date_range(start=start_date, end=end_date)

# Create an empty DataFrame to store all stock data
all_stocks_data = pd.DataFrame()
stock_data_list = []

for ticker in z_distinct_tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.reindex(date_range)
    stock_data.fillna(0, inplace=True)
    stock_data['Ticker'] = ticker
    stock_data_list.append(stock_data)

all_stocks_data = pd.concat(stock_data_list)
all_stocks_data.reset_index(inplace=True)
all_stocks_data.rename(columns={'index': 'Date'}, inplace=True)

print(all_stocks_data)











def download_stock_data(tickers, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    stock_data_list = []

    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data = stock_data.reindex(date_range)
        stock_data.fillna(0, inplace=True)
        stock_data['Ticker'] = ticker
        stock_data_list.append(stock_data)
    all_stocks_data = pd.concat(stock_data_list)
    all_stocks_data.reset_index(inplace=True)
    all_stocks_data.rename(columns={'index': 'Date'}, inplace=True)

    return all_stocks_data



def compare_dates(df1, df2):
    dates_df1 = set(df1['Date'])
    dates_df2 = set(df2['Date'])

    unique_dates_df1 = dates_df1 - dates_df2
    unique_dates_df2 = dates_df2 - dates_df1

    return unique_dates_df1, unique_dates_df2


# List of individual tickers
z_distinct_tickers = ['AAPL', 'MSFT', 'GOOG']
start_date = '2023-12-01'
end_date = '2023-12-10'


# Download data
individual_stocks_data = download_stock_data(z_distinct_tickers, start_date, end_date)
spy_data = download_stock_data(['SPY'], start_date, end_date)
spy_data = spy_data[spy_data['Adj Close'] != 0]


# Compare dates between the two datasets
unique_dates_individual, unique_dates_spy = compare_dates(individual_stocks_data, spy_data)
print("Unique dates in individual stocks dataset:", unique_dates_individual)
print("Unique dates in SPY dataset:", unique_dates_spy)


def drop_unique_dates(df1, df2):
    dates_df1 = set(df1['Date'])
    dates_df2 = set(df2['Date'])

    unique_dates_df1 = dates_df1 - dates_df2

    return df1[~df1['Date'].isin(unique_dates_df1)]


# Drop any dates in individual_stocks_data that are unique when compared to spy_data
individual_stocks_data_updated = drop_unique_dates(individual_stocks_data, spy_data)



###################################################################
                                #THREADING
###################################################################



def download_single_stock_data(ticker, start_date, end_date, date_range):
    """
    Download stock data for a single ticker.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data = stock_data.reindex(date_range)
        stock_data.fillna(0, inplace=True)
        stock_data['Ticker'] = ticker
        return ticker, stock_data
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
        return ticker, None

def download_stock_data(tickers, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    stock_data_list = []
    error_tickers = []

    with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
        # Create a future for each ticker
        future_to_ticker = {executor.submit(download_single_stock_data, ticker, start_date, end_date, date_range): ticker for ticker in tickers}

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                _, data = future.result()
                if data is not None:
                    stock_data_list.append(data)
                else:
                    error_tickers.append(ticker)
            except Exception as e:
                error_tickers.append(ticker)
                print(f"Error downloading data for {ticker}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    all_stocks_data = pd.concat(stock_data_list, ignore_index=True)
    all_stocks_data.reset_index(drop=True, inplace=True)

    return all_stocks_data, error_tickers

# Example usage
distinct_tickers = ['GOOGL','BRK.B']
start_date = '2020-01-01'
end_date = '2023-01-01'
individual_stocks_data, failed_tickers = download_stock_data(distinct_tickers, start_date, end_date)



# Now, 'individual_stocks_data' contains the downloaded data, 
# and 'failed_tickers' contains the list of tickers that had errors during download





import requests

url = "https://backend.simfin.com/api/v3/companies/prices/compact?id=&ticker=XTO"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)

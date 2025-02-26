import pandas as pd
import requests
import os
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt

#########################################
# 1) Download & Extract COT Data
#########################################

COT_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip"
SAVE_DIR = "./cot_data"

os.makedirs(SAVE_DIR, exist_ok=True)

def download_and_extract_cot_data(start_year, end_year, save_dir):
    for year in range(start_year, end_year + 1):
        url = COT_URL_TEMPLATE.format(year=year)
        zip_path = os.path.join(save_dir, f"cot_{year}.zip")

        try:
            print(f"Downloading data for {year}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                
                # Extract the ZIP 
                with zipfile.ZipFile(zip_path, "r") as z:
                    file_list = z.namelist()
                    print(f"Contents of {year}.zip: {file_list}")  # Debugging
                    # Attempt to extract yearly file if present
                    expected_file = f"com_disagg_txt_{year}.txt"
                    
                    if expected_file in file_list:
                        print(f"Extracting {expected_file} for {year}...")
                        extracted_path = z.extract(expected_file, save_dir)
                    elif "c_year.txt" in file_list:
                        print(f"Extracting fallback file c_year.txt for {year}...")
                        extracted_path = z.extract("c_year.txt", save_dir)
                        # Rename so each year's c_year doesn't overwrite
                        new_filename = f"cot_{year}.txt"
                        new_path = os.path.join(save_dir, new_filename)
                        os.rename(os.path.join(save_dir,"c_year.txt"), new_path)
                        print(f"Renamed c_year.txt to {new_filename}")
                    else:
                        print(f"No usable file found in {year}.zip.")
                print(f"Data for {year} downloaded and extracted.")
            else:
                print(f"No data available for {year} (HTTP {response.status_code}).")
        except Exception as e:
            print(f"Failed to download/extract data for {year}: {e}")

def process_data_for_all_years(save_dir, start_year, end_year):
    all_data = []

    # See how many text files are now in the folder
    extracted_files = [f for f in os.listdir(save_dir) if f.endswith(".txt")]
    print(f"\nExtracted text files (count): {len(extracted_files)}")
    for name in extracted_files[:10]:
        print("  -", name)  # show first 10 file names

    # Read each .txt into a DataFrame, then combine
    for file in extracted_files:
        file_path = os.path.join(save_dir, file)
        print(f"Reading file: {file_path} ...")
        try:
            df = pd.read_csv(file_path, sep=",", engine="python", on_bad_lines="skip")
            all_data.append(df)
        except Exception as e:
            print(f"Could not parse {file_path}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()

#########################################
# 2) Main: Download, Combine, Filter
#########################################

start_year = 2010
end_year   = datetime.now().year

# 2A) Download data 
download_and_extract_cot_data(start_year, end_year, SAVE_DIR)

# 2B) Combine all data into one DataFrame
cot_data = process_data_for_all_years(SAVE_DIR, start_year, end_year)

# Debug Print #1: Check if any data
if cot_data.empty:
    print("No data found at all. Exiting.")
    exit()
print("Combined data shape:", cot_data.shape)

# Debug Print #2: Commodity names
if 'Market_and_Exchange_Names' in cot_data.columns:
    unique_commodities = cot_data['Market_and_Exchange_Names'].unique()
    print("\nUnique commodity names in data (first 50 shown):")
    for c in unique_commodities[:50]:
        print("   ", c)
else:
    print("Column 'Market_and_Exchange_Names' not found in combined data.")
    exit()

# Debug Print #3: Check columns
print("\nColumns in combined data:")
print(cot_data.columns)

# Just LIVE CATTLE
commodity_name = "LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE"
print(f"\nFiltering for commodity: {commodity_name}")

# 2C) Filter for Live Cattle rows & copy
temp_df = cot_data[cot_data['Market_and_Exchange_Names'].str.contains(commodity_name, na=False)]
cattle_df = temp_df.copy()  # avoid SettingWithCopyWarning

# Debug Print #4: Check shape & head
print("Number of rows for cattle_df after commodity filter:", cattle_df.shape)
print(cattle_df.head(5))

if cattle_df.empty:
    print(f"No data found for {commodity_name}. Exiting.")
    exit()

#########################################
# 3) Compute Z-Scores for MM Long & Short
#########################################

# A) Identify date column and parse it
if 'As_of_Date_In_Form_YYMMDD' in cattle_df.columns:
    date_col = 'As_of_Date_In_Form_YYMMDD'
    cattle_df['Date'] = pd.to_datetime(
        cattle_df[date_col].astype(str), 
        format='%y%m%d', 
        errors='coerce'
    )
elif 'As_of_Date' in cattle_df.columns:
    date_col = 'As_of_Date'
    cattle_df['Date'] = pd.to_datetime(
        cattle_df[date_col], 
        errors='coerce'
    )
else:
    print("No recognized date column found for date parsing. Exiting.")
    exit()

# B) Drop rows with invalid or missing Date
before_drop = len(cattle_df)
cattle_df.dropna(subset=['Date'], inplace=True)
after_drop = len(cattle_df)
print(f"\nDropped {before_drop - after_drop} rows due to invalid Date.")
print("Now have", after_drop, "rows.")

# Debug Print #5: Check date parse result
print("First few Date values after parse:")
print(cattle_df[['Date']].head(5))

# C) Sort by date (ascending)
cattle_df.sort_values('Date', ascending=True, inplace=True)

long_col  = "M_Money_Positions_Long_All"
short_col = "M_Money_Positions_Short_All"

# Debug Print #6: Check if these columns exist
for col in [long_col, short_col]:
    if col not in cattle_df.columns:
        print(f"Missing column '{col}'. Exiting.")
        exit()


window_size = 26

def compute_zscore(series, window=26):
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std  = series.rolling(window=window, min_periods=1).std()
    return (series - rolling_mean) / rolling_std

cattle_df['ZScore_Long']  = compute_zscore(cattle_df[long_col], window=window_size)
cattle_df['ZScore_Short'] = compute_zscore(cattle_df[short_col], window=window_size)

# Additional debug: earliest & latest date in cattle_df
print("\nEarliest date in cattle_df:", cattle_df['Date'].min())
print("Latest date in cattle_df:", cattle_df['Date'].max())

# Debug Print #7: Check Z-score values
print("\nZ-scores sample:")
print(cattle_df[['Date','ZScore_Long','ZScore_Short']].head(10))

#########################################
# 4) Plot
#########################################

# A) Set 'Date' as the index
cattle_df.set_index('Date', inplace=True)

#########################################
# 4B) Filter to Last 14 Months & Plot
#########################################

# 4B-1) Find newest date
max_date = cattle_df.index.max()
print("Most recent date in dataset:", max_date)

# 4B-2) Compute cutoff date for 14 months ago
cutoff_date = max_date - pd.DateOffset(months=14)
print("Cutoff date (14 months ago):", cutoff_date)

# 4B-3) Filter DataFrame to that date range
filtered_df = cattle_df.loc[cattle_df.index >= cutoff_date]

if filtered_df.empty:
    print("No data in the last 14 months to plot.")
else:
    ax = filtered_df[['ZScore_Short','ZScore_Long']].plot(
        figsize=(10,6),
        title="Z Score Short vs. Z Score Long (Managed Money) - Live Cattle (Last 14 Months)"
    )
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Z Score")
    plt.tight_layout()
    plt.show()

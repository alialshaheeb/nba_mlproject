import pandas as pd
import time

# Scrape 2010-2024 advanced stats
dfs = []
for year in range(2010, 2025):
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        tables = pd.read_html(url)
        df = tables[0]
        df['Season'] = year
        df['Season_Label'] = f"{year-1}-{str(year)[2:]}"
        dfs.append(df)
        print(f"Got {year}")
        time.sleep(4)
    except Exception as e:
        print(f"Failed {year}: {e}")

# Load your existing 2025 data from data folder
existing = pd.read_csv('data/advanced_stats.csv', skiprows=1)
existing['Season'] = 2025
existing['Season_Label'] = "2024-25"

# Combine everything
new_data = pd.concat(dfs, ignore_index=True)
all_data = pd.concat([new_data, existing], ignore_index=True)

# Save back to data folder
all_data.to_csv('data/advanced_stats_all.csv', index=False)
print("Done:", all_data.shape)
print(all_data[['Player', 'Season', 'Season_Label']].head(20))
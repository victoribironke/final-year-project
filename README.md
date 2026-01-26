# How to run the data module

cd data

## Full pipeline (fetches weather - takes ~15-20 min first run)
python main.py

## Skip weather fetch (use cached data)
python main.py --skip-weather

## Export harmonized data to CSV
python main.py --export-csv harmonized_food_prices.csv
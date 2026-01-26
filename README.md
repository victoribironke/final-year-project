## How to run the data module
```bash
cd data
```

### Full pipeline (fetches weather - takes ~15-20 min first run)
```bash
python main.py
```

### Skip weather fetch (use cached data)
```bash
python main.py --skip-weather
```
### Export harmonized data to CSV
```bash
python main.py --export-csv harmonized_food_prices.csv
```
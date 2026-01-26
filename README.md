## How to run the data module
```python
cd data
```

### Full pipeline (fetches weather - takes ~15-20 min first run)
```python
python main.py
```

### Skip weather fetch (use cached data)
```python
python main.py --skip-weather
```
### Export harmonized data to CSV
```python
python main.py --export-csv harmonized_food_prices.csv
```
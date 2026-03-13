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

### Full Research Documentation
[Google Docs](https://docs.google.com/document/d/1kclR0sXW6X9UQ1pFBR9JZwS7BQr2g1OT1AunKlmQ5xA/edit?usp=drivesdk)
[Google Colab](https://colab.research.google.com/drive/16ViKbvAMM-7CnuQqF_bFsrCceQKXPei4?usp=sharing)
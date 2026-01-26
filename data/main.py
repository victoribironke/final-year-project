"""
Main entry point for the WFP Food Price Data Pipeline.

This module processes raw CSV data from the World Food Programme,
enriches it with historical weather data, and stores the harmonized
dataset in a SQLite database.

Target Schema:
- Date: The date of the data record (Datetime)
- Demand: Estimated from price (0-100 scale, higher price = higher demand) (Numerical)
- Commodity_Name: The name of the crop (Categorical)
- Market_Location: The market or region (Categorical)
- Unit_Price: The average price of the commodity (Numerical)
- Avg_Temperature: Average temperature for the region (Numerical)
- Rainfall: Rainfall measurement for the region (Numerical)
- Is_Holiday: Binary indicator for Nigerian public holidays (Binary)

Usage:
    # Run the full pipeline
    python main.py
    
    # Skip weather fetching (use cached data)
    python main.py --skip-weather
    
    # Export to CSV after processing
    python main.py --export-csv harmonized_data.csv
"""

from pipeline import DataPipeline, main as run_pipeline


def quick_start():
    """
    Quick start function for running the pipeline with default settings.
    Useful for testing or running from an IDE.
    """
    print("=" * 60)
    print("WFP Food Price Data Pipeline - Quick Start")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Read the WFP food prices CSV")
    print("  2. Fetch historical weather data from Open-Meteo API (free)")
    print("  3. Estimate demand from price (normalized 0-100 per commodity)")
    print("  4. Store harmonized data in SQLite database")
    print("\nNote: Weather fetching may take time on first run")
    print("      (~63 locations × date ranges = ~63 API calls)")
    print("=" * 60)
    
    # Create and run pipeline with default settings
    pipeline = DataPipeline(
        csv_path="wfp_food_prices_nga.csv",
        db_path="food_prices.db"
    )
    
    results = pipeline.run()
    
    # Optionally export to CSV
    print("\nDo you want to export to CSV? (The data is already in SQLite)")
    
    return results


if __name__ == "__main__":
    import sys
    
    # If run with arguments, use the CLI
    if len(sys.argv) > 1:
        run_pipeline()
    else:
        # Otherwise run quick start
        quick_start()

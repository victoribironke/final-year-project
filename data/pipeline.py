"""
Main data processing pipeline.
Orchestrates CSV processing, weather fetching, and database population.
"""

import os
from datetime import datetime, date
from typing import Optional, Dict, Tuple, List
from collections import defaultdict
from tqdm import tqdm

from models import DatabaseManager, FoodPriceRecord, WeatherCache
from csv_processor import CSVProcessor, RawPriceRecord
from weather_api import OpenMeteoClient, WeatherData
from holidays import HOLIDAY_CACHE, is_holiday


class DemandEstimator:
    """
    Estimates demand based on price.
    
    Since price is directly proportional to demand (higher demand = higher price),
    we normalize prices within each commodity to create a demand index.
    
    Formula: demand = (price - min_price) / (max_price - min_price) * 100
    This gives a 0-100 scale where 100 = highest demand period.
    """
    
    def __init__(self):
        self.commodity_stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, records: List[RawPriceRecord]):
        """
        Calculate min/max prices per commodity from all records.
        
        Args:
            records: List of raw price records
        """
        # Group prices by commodity
        commodity_prices = defaultdict(list)
        for r in records:
            commodity_prices[r.commodity].append(r.price)
        
        # Calculate stats for each commodity
        for commodity, prices in commodity_prices.items():
            self.commodity_stats[commodity] = {
                'min': min(prices),
                'max': max(prices),
                'mean': sum(prices) / len(prices)
            }
    
    def estimate_demand(self, commodity: str, price: float) -> Optional[float]:
        """
        Estimate demand index (0-100) based on price.
        
        Args:
            commodity: The commodity name
            price: The current price
            
        Returns:
            Demand index (0-100) or None if commodity unknown
        """
        if commodity not in self.commodity_stats:
            return None
        
        stats = self.commodity_stats[commodity]
        min_price = stats['min']
        max_price = stats['max']
        
        # Avoid division by zero
        if max_price == min_price:
            return 50.0  # Neutral demand if no price variation
        
        # Normalize to 0-100 scale
        demand = ((price - min_price) / (max_price - min_price)) * 100
        
        # Clamp to valid range (in case of outliers)
        return max(0.0, min(100.0, demand))


class DataPipeline:
    """
    Main pipeline for processing food price data with weather enrichment.
    """
    
    def __init__(self, 
                 csv_path: str,
                 db_path: str = "food_prices.db"):
        """
        Initialize the data pipeline.
        
        Args:
            csv_path: Path to the WFP CSV file
            db_path: Path to the SQLite database
        """
        self.csv_processor = CSVProcessor(csv_path)
        self.db_manager = DatabaseManager(db_path)
        self.weather_client = OpenMeteoClient()
        self.demand_estimator = DemandEstimator()
        
        # Initialize database
        self.db_manager.create_tables()
    
    def _get_cached_weather(self, session, lat: float, lon: float, 
                            target_date: date) -> Optional[WeatherCache]:
        """Check if weather data is already cached in the database."""
        return session.query(WeatherCache).filter(
            WeatherCache.date == datetime.combine(target_date, datetime.min.time()),
            WeatherCache.latitude == round(lat, 2),
            WeatherCache.longitude == round(lon, 2)
        ).first()
    
    def _cache_weather(self, session, weather_data: WeatherData):
        """Cache weather data in the database."""
        cache_entry = WeatherCache(
            date=datetime.combine(weather_data.date, datetime.min.time()),
            latitude=round(weather_data.latitude, 2),
            longitude=round(weather_data.longitude, 2),
            avg_temperature=weather_data.avg_temperature,
            rainfall=weather_data.rainfall,
            api_source="open-meteo"
        )
        session.merge(cache_entry)
    
    def fetch_and_cache_weather(self, batch_size: int = 100) -> int:
        """
        Fetch weather data for all unique location-date combinations and cache it.
        Uses efficient batch fetching per location.
        
        Args:
            batch_size: Number of locations to process before committing
            
        Returns:
            Number of weather records fetched
        """
        session = self.db_manager.get_session()
        
        # Get dates grouped by location for efficient batch fetching
        location_dates = self.csv_processor.get_dates_by_location()
        
        print(f"Found {len(location_dates)} unique locations to fetch weather for")
        
        total_fetched = 0
        total_cached = 0
        
        try:
            for i, ((lat, lon), dates) in enumerate(tqdm(location_dates.items(), 
                                                          desc="Fetching weather")):
                # Check which dates are not yet cached
                uncached_dates = []
                for d in dates:
                    if not self._get_cached_weather(session, lat, lon, d):
                        uncached_dates.append(d)
                    else:
                        total_cached += 1
                
                if not uncached_dates:
                    continue
                
                # Fetch weather for uncached dates using batch API
                weather_dict = self.weather_client.get_weather_batch_for_location(
                    lat, lon, uncached_dates
                )
                
                # Cache the results
                for target_date, weather in weather_dict.items():
                    self._cache_weather(session, weather)
                    total_fetched += 1
                
                # Commit periodically
                if (i + 1) % batch_size == 0:
                    session.commit()
                    print(f"Progress: {i + 1}/{len(location_dates)} locations, "
                          f"{total_fetched} fetched, {total_cached} already cached")
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"Error fetching weather: {e}")
            raise
        finally:
            session.close()
        
        print(f"\nWeather fetch complete:")
        print(f"  New records fetched: {total_fetched}")
        print(f"  Already cached: {total_cached}")
        
        return total_fetched
    
    def populate_database(self, batch_size: int = 1000, clear_existing: bool = True) -> int:
        """
        Process all CSV records and populate the database with harmonized data.
        
        Args:
            batch_size: Number of records to process before committing
            clear_existing: If True, clear existing food price records before inserting
            
        Returns:
            Number of records inserted
        """
        session = self.db_manager.get_session()
        
        # Clear existing records if requested
        if clear_existing:
            existing_count = session.query(FoodPriceRecord).count()
            if existing_count > 0:
                print(f"Clearing {existing_count:,} existing records...")
                session.query(FoodPriceRecord).delete()
                session.commit()
        
        # Load all records first for demand estimation
        print("Loading records for demand estimation...")
        all_records = list(self.csv_processor.read_records())
        
        # Fit the demand estimator on all data
        print("Fitting demand estimator...")
        self.demand_estimator.fit(all_records)
        print(f"  Computed stats for {len(self.demand_estimator.commodity_stats)} commodities")
        
        # Build weather cache lookup
        weather_cache: Dict[Tuple[date, float, float], Tuple[float, float]] = {}
        for cache_entry in session.query(WeatherCache).all():
            key = (cache_entry.date.date(), 
                   round(cache_entry.latitude, 2), 
                   round(cache_entry.longitude, 2))
            weather_cache[key] = (cache_entry.avg_temperature, cache_entry.rainfall)
        
        print(f"Loaded {len(weather_cache)} weather cache entries")
        
        records_inserted = 0
        records_skipped = 0
        
        try:
            for i, raw_record in enumerate(tqdm(all_records, desc="Processing records")):
                # Look up weather data
                weather_key = (raw_record.date, 
                              round(raw_record.latitude, 2),
                              round(raw_record.longitude, 2))
                
                avg_temp, rainfall = weather_cache.get(weather_key, (None, None))
                
                # Create market location string
                market_location = f"{raw_record.market}, {raw_record.lga}, {raw_record.state}"
                
                # Check if holiday
                holiday = is_holiday(raw_record.date, HOLIDAY_CACHE)
                
                # Estimate demand based on price
                demand = self.demand_estimator.estimate_demand(
                    raw_record.commodity, raw_record.price
                )
                
                # Create harmonized record
                record = FoodPriceRecord(
                    # Core schema fields
                    date=datetime.combine(raw_record.date, datetime.min.time()),
                    demand=demand,
                    commodity_name=raw_record.commodity,
                    market_location=market_location,
                    unit_price=raw_record.price,
                    avg_temperature=avg_temp,
                    rainfall=rainfall,
                    is_holiday=holiday,
                    
                    # Additional source fields
                    state=raw_record.state,
                    lga=raw_record.lga,
                    latitude=raw_record.latitude,
                    longitude=raw_record.longitude,
                    commodity_category=raw_record.category,
                    unit=raw_record.unit,
                    price_type=raw_record.price_type,
                    currency=raw_record.currency,
                    price_usd=raw_record.price_usd
                )
                
                try:
                    session.add(record)
                    records_inserted += 1
                except Exception:
                    records_skipped += 1
                
                # Commit periodically
                if (i + 1) % batch_size == 0:
                    try:
                        session.commit()
                    except Exception as e:
                        session.rollback()
                        print(f"Batch commit error: {e}")
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"Error populating database: {e}")
            raise
        finally:
            session.close()
        
        print(f"\nDatabase population complete:")
        print(f"  Records inserted: {records_inserted}")
        print(f"  Records skipped: {records_skipped}")
        
        return records_inserted
    
    def run(self, skip_weather: bool = False) -> Dict:
        """
        Run the complete data pipeline.
        
        Args:
            skip_weather: If True, skip weather fetching (use existing cache)
            
        Returns:
            Dictionary with pipeline statistics
        """
        print("=" * 60)
        print("Starting Data Pipeline")
        print("=" * 60)
        
        # Step 1: Show CSV statistics
        print("\n[Step 1] Analyzing CSV file...")
        stats = self.csv_processor.get_statistics()
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Unique locations: {stats['unique_locations']}")
        print(f"  Unique dates: {stats['unique_dates']}")
        print(f"  Location-date combinations: {stats['unique_location_dates']}")
        print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"  States: {len(stats['states'])}")
        print(f"  Commodities: {len(stats['commodities'])}")
        
        # Step 2: Fetch weather data
        weather_fetched = 0
        if not skip_weather:
            print("\n[Step 2] Fetching weather data from Open-Meteo...")
            print("(This may take a while for first run)")
            weather_fetched = self.fetch_and_cache_weather()
        else:
            print("\n[Step 2] Skipping weather fetch (using existing cache)")
        
        # Step 3: Populate database
        print("\n[Step 3] Populating database with harmonized records...")
        print("(Demand is estimated from price: higher price = higher demand)")
        records_inserted = self.populate_database()
        
        # Summary
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        
        return {
            'csv_stats': stats,
            'weather_fetched': weather_fetched,
            'records_inserted': records_inserted
        }
    
    def export_to_csv(self, output_path: str = "harmonized_food_prices.csv"):
        """
        Export the harmonized data from the database to a CSV file.
        
        Args:
            output_path: Path for the output CSV file
        """
        import csv
        
        session = self.db_manager.get_session()
        
        try:
            records = session.query(FoodPriceRecord).all()
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header matching target schema
                writer.writerow([
                    'Date', 'Demand', 'Commodity_Name', 'Market_Location',
                    'Unit_Price', 'Avg_Temperature', 'Rainfall', 'Is_Holiday',
                    'State', 'LGA', 'Latitude', 'Longitude', 'Category',
                    'Unit', 'Price_Type', 'Currency', 'Price_USD'
                ])
                
                for r in records:
                    writer.writerow([
                        r.date.strftime('%Y-%m-%d') if r.date else '',
                        round(r.demand, 2) if r.demand is not None else '',
                        r.commodity_name,
                        r.market_location,
                        r.unit_price,
                        r.avg_temperature if r.avg_temperature is not None else '',
                        r.rainfall if r.rainfall is not None else '',
                        1 if r.is_holiday else 0,
                        r.state,
                        r.lga,
                        r.latitude,
                        r.longitude,
                        r.commodity_category,
                        r.unit,
                        r.price_type,
                        r.currency,
                        r.price_usd
                    ])
            
            print(f"Exported {len(records)} records to {output_path}")
            
        finally:
            session.close()


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Food Price Data Pipeline")
    parser.add_argument(
        "--csv", 
        default="wfp_food_prices_nga.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--db",
        default="food_prices.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather fetching (use existing cache)"
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Export harmonized data to CSV after processing"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = DataPipeline(
        csv_path=args.csv,
        db_path=args.db
    )
    
    results = pipeline.run(skip_weather=args.skip_weather)
    
    # Export if requested
    if args.export_csv:
        pipeline.export_to_csv(args.export_csv)
    
    return results


if __name__ == "__main__":
    main()

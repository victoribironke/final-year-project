"""
CSV processor for WFP food prices data.
Extracts unique date/location combinations and processes records.
"""

import csv
from datetime import datetime, date
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RawPriceRecord:
    """Raw record from WFP CSV file."""
    date: date
    state: str           # admin1
    lga: str             # admin2
    market: str
    market_id: str
    latitude: float
    longitude: float
    category: str
    commodity: str
    commodity_id: str
    unit: str
    price_flag: str
    price_type: str
    currency: str
    price: float
    price_usd: float


@dataclass 
class LocationDateKey:
    """Unique key for weather data lookup."""
    date: date
    latitude: float
    longitude: float
    
    def __hash__(self):
        # Round coordinates to 2 decimal places for grouping nearby locations
        return hash((self.date, round(self.latitude, 2), round(self.longitude, 2)))
    
    def __eq__(self, other):
        if not isinstance(other, LocationDateKey):
            return False
        return (self.date == other.date and 
                round(self.latitude, 2) == round(other.latitude, 2) and
                round(self.longitude, 2) == round(other.longitude, 2))


class CSVProcessor:
    """Processes WFP food price CSV files."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the CSV processor.
        
        Args:
            csv_path: Path to the WFP CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    def _parse_row(self, row: Dict) -> RawPriceRecord:
        """Parse a CSV row into a RawPriceRecord."""
        return RawPriceRecord(
            date=datetime.strptime(row['date'], '%Y-%m-%d').date(),
            state=row['admin1'].strip(),
            lga=row['admin2'].strip(),
            market=row['market'].strip(),
            market_id=row['market_id'].strip(),
            latitude=float(row['latitude']),
            longitude=float(row['longitude']),
            category=row['category'].strip(),
            commodity=row['commodity'].strip(),
            commodity_id=row['commodity_id'].strip(),
            unit=row['unit'].strip(),
            price_flag=row['priceflag'].strip(),
            price_type=row['pricetype'].strip(),
            currency=row['currency'].strip(),
            price=float(row['price']),
            price_usd=float(row['usdprice'])
        )
    
    def read_records(self) -> Iterator[RawPriceRecord]:
        """
        Read all records from the CSV file.
        
        Yields:
            RawPriceRecord for each valid row
        """
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Skip the HXL tag row (starts with #)
                if row.get('date', '').startswith('#'):
                    continue
                    
                try:
                    yield self._parse_row(row)
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid row: {e}")
                    continue
    
    def get_unique_location_dates(self) -> Set[LocationDateKey]:
        """
        Extract all unique (date, latitude, longitude) combinations.
        
        Returns:
            Set of LocationDateKey objects
        """
        unique_keys = set()
        
        for record in self.read_records():
            key = LocationDateKey(
                date=record.date,
                latitude=record.latitude,
                longitude=record.longitude
            )
            unique_keys.add(key)
        
        return unique_keys
    
    def get_locations_by_date(self) -> Dict[date, List[Tuple[float, float]]]:
        """
        Group unique locations by date for efficient batch weather fetching.
        
        Returns:
            Dictionary mapping dates to lists of (lat, lon) tuples
        """
        date_locations = defaultdict(set)
        
        for record in self.read_records():
            lat_lon = (round(record.latitude, 2), round(record.longitude, 2))
            date_locations[record.date].add(lat_lon)
        
        return {d: list(locs) for d, locs in date_locations.items()}
    
    def get_dates_by_location(self) -> Dict[Tuple[float, float], List[date]]:
        """
        Group dates by location for efficient batch weather fetching.
        This is optimal for Open-Meteo which supports date ranges.
        
        Returns:
            Dictionary mapping (lat, lon) tuples to lists of dates
        """
        location_dates = defaultdict(set)
        
        for record in self.read_records():
            lat_lon = (round(record.latitude, 2), round(record.longitude, 2))
            location_dates[lat_lon].add(record.date)
        
        return {loc: sorted(list(dates)) for loc, dates in location_dates.items()}
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the CSV file.
        
        Returns:
            Dictionary with various statistics
        """
        records = list(self.read_records())
        
        unique_locations = set()
        unique_dates = set()
        unique_commodities = set()
        unique_markets = set()
        states = set()
        
        for r in records:
            unique_locations.add((round(r.latitude, 2), round(r.longitude, 2)))
            unique_dates.add(r.date)
            unique_commodities.add(r.commodity)
            unique_markets.add(r.market)
            states.add(r.state)
        
        date_range = (min(unique_dates), max(unique_dates))
        
        return {
            'total_records': len(records),
            'unique_locations': len(unique_locations),
            'unique_dates': len(unique_dates),
            'unique_location_dates': len(self.get_unique_location_dates()),
            'unique_commodities': len(unique_commodities),
            'commodities': sorted(unique_commodities),
            'unique_markets': len(unique_markets),
            'states': sorted(states),
            'date_range': date_range,
        }


if __name__ == "__main__":
    # Test CSV processor
    processor = CSVProcessor("wfp_food_prices_nga.csv")
    
    stats = processor.get_statistics()
    print("CSV Statistics:")
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Unique locations: {stats['unique_locations']}")
    print(f"  Unique dates: {stats['unique_dates']}")
    print(f"  Unique location-date combinations: {stats['unique_location_dates']}")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"  States: {len(stats['states'])}")
    print(f"  Commodities: {len(stats['commodities'])}")
    print(f"\nCommodities: {stats['commodities'][:10]}...")

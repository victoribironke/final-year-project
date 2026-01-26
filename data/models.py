"""
Database models and schema definitions for the harmonized food price dataset.
Uses SQLAlchemy for ORM and SQLite for storage.
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, UniqueConstraint, Index
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class FoodPriceRecord(Base):
    """
    Harmonized food price record conforming to the target schema:
    - Date: The date of the data record
    - Demand (Target): Market sales volume (not available in source, set to NULL)
    - Commodity_Name: The name of the crop
    - Market_Location: The market or region
    - Unit_Price: The average price of the commodity
    - Avg_Temperature: Average temperature for the region
    - Rainfall: Rainfall measurement for the region
    - Is_Holiday: Binary indicator for public holidays
    """
    __tablename__ = 'food_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields from target schema
    date = Column(DateTime, nullable=False, index=True)
    demand = Column(Float, nullable=True)  # Not available in WFP data
    commodity_name = Column(String(100), nullable=False, index=True)
    market_location = Column(String(200), nullable=False, index=True)
    unit_price = Column(Float, nullable=False)
    avg_temperature = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)
    is_holiday = Column(Boolean, default=False)
    
    # Additional fields from source data (useful for analysis)
    state = Column(String(100), nullable=True)  # admin1
    lga = Column(String(100), nullable=True)    # admin2
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    commodity_category = Column(String(100), nullable=True)
    unit = Column(String(50), nullable=True)
    price_type = Column(String(50), nullable=True)  # Retail/Wholesale
    currency = Column(String(10), nullable=True)
    price_usd = Column(Float, nullable=True)
    
    # Index for efficient querying (no unique constraint - source data has duplicates)
    __table_args__ = (
        Index('ix_price_lookup', 'date', 'commodity_name', 'market_location'),
    )


class WeatherCache(Base):
    """
    Cache for weather API responses to avoid redundant API calls.
    Stores weather data by date and location.
    """
    __tablename__ = 'weather_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    avg_temperature = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    api_source = Column(String(50), nullable=True)  # 'open-meteo'
    
    __table_args__ = (
        UniqueConstraint('date', 'latitude', 'longitude', name='uix_weather_location_date'),
    )


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, db_path: str = "food_prices.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """Get a new database session."""
        return self.Session()
    
    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(self.engine)


if __name__ == "__main__":
    # Test database creation
    db = DatabaseManager()
    db.create_tables()
    print("Database tables created successfully!")

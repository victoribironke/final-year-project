"""
Weather API clients for fetching historical weather data.
Primary: Open-Meteo (free, no API key required)
Fallback: Weatherbit.io (requires API key)
"""

import time
import requests
from datetime import datetime, date
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class WeatherData:
    """Weather data for a specific date and location."""
    date: date
    latitude: float
    longitude: float
    avg_temperature: Optional[float]  # Celsius
    rainfall: Optional[float]  # mm
    source: str


class OpenMeteoClient:
    """
    Client for Open-Meteo Historical Weather API.
    Free, no API key required, data from 1940 onwards.
    https://open-meteo.com/en/docs/historical-weather-api
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, requests_per_second: float = 5.0):
        """
        Initialize the Open-Meteo client.
        
        Args:
            requests_per_second: Rate limit (default 5 req/s to be respectful)
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
    
    def get_weather(self, lat: float, lon: float, target_date: date) -> Optional[WeatherData]:
        """
        Fetch historical weather data for a specific date and location.
        
        Args:
            lat: Latitude
            lon: Longitude  
            target_date: The date to fetch weather for
            
        Returns:
            WeatherData object or None if request fails
        """
        self._rate_limit()
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Africa/Lagos"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            temps = daily.get("temperature_2m_mean", [None])
            precip = daily.get("precipitation_sum", [None])
            
            return WeatherData(
                date=target_date,
                latitude=lat,
                longitude=lon,
                avg_temperature=temps[0] if temps else None,
                rainfall=precip[0] if precip else None,
                source="open-meteo"
            )
            
        except requests.exceptions.RequestException as e:
            print(f"Open-Meteo API error for {target_date} at ({lat}, {lon}): {e}")
            return None
    
    def get_weather_batch(self, lat: float, lon: float, 
                          start_date: date, end_date: date) -> list[WeatherData]:
        """
        Fetch weather data for a date range (more efficient for bulk operations).
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of WeatherData objects
        """
        self._rate_limit()
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Africa/Lagos"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            temps = daily.get("temperature_2m_mean", [])
            precip = daily.get("precipitation_sum", [])
            
            results = []
            for i, date_str in enumerate(dates):
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                results.append(WeatherData(
                    date=d,
                    latitude=lat,
                    longitude=lon,
                    avg_temperature=temps[i] if i < len(temps) else None,
                    rainfall=precip[i] if i < len(precip) else None,
                    source="open-meteo"
                ))
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Open-Meteo batch API error: {e}")
            return []


class WeatherbitClient:
    """
    Client for Weatherbit Historical Weather API (fallback).
    Requires API key, limited free tier (50 calls/day).
    https://www.weatherbit.io/api/historical-weather-daily
    """
    
    BASE_URL = "https://api.weatherbit.io/v2.0/history/daily"
    
    def __init__(self, api_key: str, requests_per_second: float = 1.0):
        """
        Initialize the Weatherbit client.
        
        Args:
            api_key: Weatherbit API key
            requests_per_second: Rate limit (default 1 req/s for free tier)
        """
        self.api_key = api_key
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
    
    def get_weather(self, lat: float, lon: float, target_date: date) -> Optional[WeatherData]:
        """
        Fetch historical weather data for a specific date and location.
        
        Args:
            lat: Latitude
            lon: Longitude
            target_date: The date to fetch weather for
            
        Returns:
            WeatherData object or None if request fails
        """
        self._rate_limit()
        
        # Weatherbit requires start_date and end_date
        params = {
            "lat": lat,
            "lon": lon,
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "key": self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("data"):
                day_data = data["data"][0]
                return WeatherData(
                    date=target_date,
                    latitude=lat,
                    longitude=lon,
                    avg_temperature=day_data.get("temp"),
                    rainfall=day_data.get("precip"),
                    source="weatherbit"
                )
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Weatherbit API error for {target_date} at ({lat}, {lon}): {e}")
            return None


class WeatherService:
    """
    Unified weather service that tries Open-Meteo first, falls back to Weatherbit.
    """
    
    def __init__(self, weatherbit_api_key: Optional[str] = None):
        """
        Initialize the weather service.
        
        Args:
            weatherbit_api_key: Optional Weatherbit API key for fallback
        """
        self.open_meteo = OpenMeteoClient()
        self.weatherbit = WeatherbitClient(weatherbit_api_key) if weatherbit_api_key else None
    
    def get_weather(self, lat: float, lon: float, target_date: date) -> Optional[WeatherData]:
        """
        Fetch weather data, trying Open-Meteo first then Weatherbit.
        
        Args:
            lat: Latitude
            lon: Longitude
            target_date: The date to fetch weather for
            
        Returns:
            WeatherData object or None if all sources fail
        """
        # Try Open-Meteo first (free, unlimited)
        result = self.open_meteo.get_weather(lat, lon, target_date)
        if result and (result.avg_temperature is not None or result.rainfall is not None):
            return result
        
        # Fall back to Weatherbit if available
        if self.weatherbit:
            result = self.weatherbit.get_weather(lat, lon, target_date)
            if result:
                return result
        
        return None
    
    def get_weather_batch_for_location(self, lat: float, lon: float,
                                        dates: list[date]) -> dict[date, WeatherData]:
        """
        Efficiently fetch weather for multiple dates at one location.
        Groups dates into ranges for batch API calls.
        
        Args:
            lat: Latitude
            lon: Longitude
            dates: List of dates to fetch weather for
            
        Returns:
            Dictionary mapping dates to WeatherData
        """
        if not dates:
            return {}
        
        sorted_dates = sorted(dates)
        min_date = sorted_dates[0]
        max_date = sorted_dates[-1]
        
        # Fetch the entire range (more efficient than individual calls)
        all_weather = self.open_meteo.get_weather_batch(lat, lon, min_date, max_date)
        
        # Create lookup dict
        weather_dict = {w.date: w for w in all_weather}
        
        # Return only the requested dates
        return {d: weather_dict[d] for d in dates if d in weather_dict}


if __name__ == "__main__":
    # Test the API clients
    client = OpenMeteoClient()
    
    # Test single date fetch
    test_date = date(2020, 6, 15)
    result = client.get_weather(9.0820, 8.6753, test_date)  # Abuja, Nigeria
    
    if result:
        print(f"Weather for Abuja on {test_date}:")
        print(f"  Temperature: {result.avg_temperature}°C")
        print(f"  Rainfall: {result.rainfall}mm")
    else:
        print("Failed to fetch weather data")

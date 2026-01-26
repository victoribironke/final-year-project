"""
Nigerian public holidays from 2002-2026.
Includes fixed holidays and approximate dates for Islamic holidays.
"""

from datetime import date
from typing import Set

# Fixed Nigerian public holidays (month, day)
FIXED_HOLIDAYS = [
    (1, 1),   # New Year's Day
    (5, 1),   # Workers' Day
    (6, 12),  # Democracy Day (changed from May 29 to June 12 in 2018)
    (10, 1),  # Independence Day
    (12, 25), # Christmas Day
    (12, 26), # Boxing Day
]

# Pre-2018, Democracy Day was May 29
DEMOCRACY_DAY_OLD = (5, 29)

# Islamic holidays (approximate dates - vary by lunar calendar)
# Eid al-Fitr (end of Ramadan) - 2 days
# Eid al-Adha (Feast of Sacrifice) - 2 days
# Mawlid (Prophet's Birthday) - 1 day

# Islamic holidays by year (approximate dates based on historical data)
# Format: {year: [(month, day), ...]}
ISLAMIC_HOLIDAYS = {
    2002: [(12, 5), (12, 6), (2, 22), (2, 23), (5, 24)],  # Eid al-Fitr, Eid al-Adha, Mawlid
    2003: [(11, 25), (11, 26), (2, 11), (2, 12), (5, 13)],
    2004: [(11, 14), (11, 15), (1, 31), (2, 1), (5, 1)],
    2005: [(11, 3), (11, 4), (1, 20), (1, 21), (4, 21)],
    2006: [(10, 23), (10, 24), (1, 9), (1, 10), (4, 10), (12, 30), (12, 31)],
    2007: [(10, 12), (10, 13), (12, 19), (12, 20), (3, 30)],
    2008: [(10, 1), (10, 2), (12, 7), (12, 8), (3, 19)],
    2009: [(9, 20), (9, 21), (11, 27), (11, 28), (3, 8)],
    2010: [(9, 9), (9, 10), (11, 16), (11, 17), (2, 25)],
    2011: [(8, 30), (8, 31), (11, 6), (11, 7), (2, 15)],
    2012: [(8, 19), (8, 20), (10, 25), (10, 26), (2, 4)],
    2013: [(8, 8), (8, 9), (10, 15), (10, 16), (1, 24)],
    2014: [(7, 28), (7, 29), (10, 4), (10, 5), (1, 13)],
    2015: [(7, 17), (7, 18), (9, 23), (9, 24), (1, 3), (12, 23)],
    2016: [(7, 6), (7, 7), (9, 11), (9, 12), (12, 11)],
    2017: [(6, 25), (6, 26), (9, 1), (9, 2), (11, 30)],
    2018: [(6, 15), (6, 16), (8, 21), (8, 22), (11, 20)],
    2019: [(6, 4), (6, 5), (8, 11), (8, 12), (11, 9)],
    2020: [(5, 24), (5, 25), (7, 31), (8, 1), (10, 29)],
    2021: [(5, 13), (5, 14), (7, 20), (7, 21), (10, 18)],
    2022: [(5, 2), (5, 3), (7, 9), (7, 10), (10, 8)],
    2023: [(4, 21), (4, 22), (6, 28), (6, 29), (9, 27)],
    2024: [(4, 10), (4, 11), (6, 16), (6, 17), (9, 15)],
    2025: [(3, 30), (3, 31), (6, 6), (6, 7), (9, 4)],
    2026: [(3, 20), (3, 21), (5, 27), (5, 28), (8, 25)],
}

# Easter holidays (Good Friday and Easter Monday)
# Format: {year: (good_friday_month, good_friday_day, easter_monday_month, easter_monday_day)}
EASTER_DATES = {
    2002: (3, 29, 4, 1),
    2003: (4, 18, 4, 21),
    2004: (4, 9, 4, 12),
    2005: (3, 25, 3, 28),
    2006: (4, 14, 4, 17),
    2007: (4, 6, 4, 9),
    2008: (3, 21, 3, 24),
    2009: (4, 10, 4, 13),
    2010: (4, 2, 4, 5),
    2011: (4, 22, 4, 25),
    2012: (4, 6, 4, 9),
    2013: (3, 29, 4, 1),
    2014: (4, 18, 4, 21),
    2015: (4, 3, 4, 6),
    2016: (3, 25, 3, 28),
    2017: (4, 14, 4, 17),
    2018: (3, 30, 4, 2),
    2019: (4, 19, 4, 22),
    2020: (4, 10, 4, 13),
    2021: (4, 2, 4, 5),
    2022: (4, 15, 4, 18),
    2023: (4, 7, 4, 10),
    2024: (3, 29, 4, 1),
    2025: (4, 18, 4, 21),
    2026: (4, 3, 4, 6),
}


def get_nigerian_holidays(year: int) -> Set[date]:
    """
    Get all Nigerian public holidays for a given year.
    
    Args:
        year: The year to get holidays for
        
    Returns:
        Set of holiday dates
    """
    holidays = set()
    
    # Fixed holidays
    for month, day in FIXED_HOLIDAYS:
        try:
            # Handle Democracy Day change in 2018
            if month == 6 and day == 12 and year < 2018:
                holidays.add(date(year, DEMOCRACY_DAY_OLD[0], DEMOCRACY_DAY_OLD[1]))
            else:
                holidays.add(date(year, month, day))
        except ValueError:
            pass  # Invalid date
    
    # Add old Democracy Day for years before 2018
    if year < 2018:
        try:
            holidays.add(date(year, 5, 29))
        except ValueError:
            pass
    
    # Easter holidays
    if year in EASTER_DATES:
        gf_m, gf_d, em_m, em_d = EASTER_DATES[year]
        holidays.add(date(year, gf_m, gf_d))  # Good Friday
        holidays.add(date(year, em_m, em_d))  # Easter Monday
    
    # Islamic holidays
    if year in ISLAMIC_HOLIDAYS:
        for month, day in ISLAMIC_HOLIDAYS[year]:
            try:
                holidays.add(date(year, month, day))
            except ValueError:
                pass
    
    return holidays


def build_holiday_cache(start_year: int = 2002, end_year: int = 2026) -> Set[date]:
    """
    Build a complete cache of all Nigerian holidays for a range of years.
    
    Args:
        start_year: First year to include
        end_year: Last year to include
        
    Returns:
        Set of all holiday dates in the range
    """
    all_holidays = set()
    for year in range(start_year, end_year + 1):
        all_holidays.update(get_nigerian_holidays(year))
    return all_holidays


def is_holiday(check_date: date, holiday_cache: Set[date] = None) -> bool:
    """
    Check if a given date is a Nigerian public holiday.
    
    Args:
        check_date: The date to check
        holiday_cache: Optional pre-built cache of holidays
        
    Returns:
        True if the date is a holiday, False otherwise
    """
    if holiday_cache:
        return check_date in holiday_cache
    return check_date in get_nigerian_holidays(check_date.year)


# Pre-built cache for quick lookups
HOLIDAY_CACHE = build_holiday_cache()


if __name__ == "__main__":
    # Test holiday functions
    print("Nigerian holidays in 2024:")
    for h in sorted(get_nigerian_holidays(2024)):
        print(f"  {h}")
    
    print(f"\nTotal holidays cached (2002-2026): {len(HOLIDAY_CACHE)}")
    
    # Test specific date
    test_date = date(2024, 10, 1)  # Independence Day
    print(f"\nIs {test_date} a holiday? {is_holiday(test_date, HOLIDAY_CACHE)}")

"""Data Integrity Dictionaries

This script stores lists and dictionaries used in the data_fetch, data_clean, and feature_engineering scripts.

No functions are present, and no libraries or modules are required.
"""

season_team_to_coach_team_dict = {
     'Brigham Young': 'BYU',
     'Cal State Long Beach': 'Long Beach State',
     'Central Connecticut State': 'Central Connecticut',
     'Central Florida': 'UCF',
     'Connecticut': 'UConn',
     'Detroit Mercy': 'Detroit',
     'East Tennessee State': 'ETSU',
     'Illinois-Chicago': 'UIC',
     'Long Island University': 'LIU',
     'Louisiana State': 'LSU',
     'Maryland-Baltimore County': 'UMBC',
     'Massachusetts': 'UMass',
     'Massachusetts-Lowell': 'UMass-Lowell',
     'Mississippi': 'Ole Miss',
     'Missouri-Kansas City': 'UMKC',
     'Nevada-Las Vegas': 'UNLV',
     'North Carolina': 'UNC',
     'North Carolina State': 'NC State',
     'North Carolina-Asheville': 'UNC Asheville',
     'North Carolina-Greensboro': 'UNC Greensboro',
     'North Carolina-Wilmington': 'UNC Wilmington',
     'Pennsylvania': 'Penn',
     'Pittsburgh': 'Pitt',
     'SIU Edwardsville': 'SIU-Edwardsville',
     "Saint Joseph's": "St. Joseph's",
     "Saint Mary's (CA)": "Saint Mary's",
     "Saint Peter's": "St. Peter's",
     'South Carolina Upstate': 'USC Upstate',
     'Southern California': 'USC',
     'Southern Methodist': 'SMU',
     'Southern Mississippi': 'Southern Miss',
     'Tennessee-Martin': 'UT-Martin',
     'Texas Christian': 'TCU',
     'Texas-El Paso': 'UTEP',
     'Texas-San Antonio': 'UTSA',
     'UC Davis': 'UC-Davis',
     'UC Irvine': 'UC-Irvine',
     'UC Santa Barbara': 'UCSB',
     'University of California': 'California',
     'Virginia Commonwealth': 'VCU',
}

coach_team_to_mm_team_dict = {
     'UAB': 'Alabama-Birmingham',
}

curr_season_to_tourney_dict = {}

rounds_str_to_numeric = {
     'Play-In': 0,
     'First Round': 1,
     'Second Round': 2,
     'Sweet Sixteen': 3,
     'Elite Eight': 4,
     'Final Four': 5,
     'National Championship': 6,
}

rounds_numeric_to_str = {value:key for (key, value) in rounds_str_to_numeric.items()}
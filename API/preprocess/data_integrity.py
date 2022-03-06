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
     'UT Arlington': 'Texas-Arlington',
}

curr_season_to_tourney_dict = {
     'Loyola (IL)': 'Loyola Chicago',
     'Norfolk State': 'Norfolk St',
     'UCSB': 'UC Santa Barbara',
     'UNC': 'North Carolina',
}

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

# def team_name_integrity_check(start_year, curr_year):
#     hist_stats_df, hist_coach_df, hist_rates_df, hist_games_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

#     for year in range(start_year, curr_year):
#         ratings = fetch.get_ratings_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-ratings.html")
#         hist_rates_df = pd.concat([hist_rates_df, ratings], ignore_index=True)        
                
#         coaches = fetch.get_coach_rankings_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-coaches.html")
#         hist_coach_df = pd.concat([hist_coach_df, coaches], ignore_index=True)
        
#         mm_games = fetch.get_hist_bracket(url=f'https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html', year=year)
#         hist_games_df = pd.concat([hist_games_df, mm_games], ignore_index=True)
        
#         stats = fetch.get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/{curr_year}-school-stats.html",
#                                      attrs={'id': 'basic_school_stats'})
#         hist_stats_df = pd.concat([hist_stats_df, stats], ignore_index=True)
     
#     school_stats_set = set(hist_stats_df['School'])
#     school_rates_set = set(hist_rates_df['Team'])
    
#     school_coach_set = set(hist_coach_df['Coach_Team'])
#     school_games_set = set(hist_games_df['Team']).union(set(hist_games_df['Team.1']))

#     rates_stats_diff = school_stats_set.difference(school_rates_set)
#     coach_rates_diff = school_rates_set.difference(school_coach_set)
#     games_coach_diff = school_coach_set.difference(school_games_set)
    
#     return rates_stats_diff, coach_rates_diff, games_coach_diff
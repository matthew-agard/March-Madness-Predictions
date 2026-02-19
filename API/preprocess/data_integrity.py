"""Data Integrity Dictionaries

This script stores lists and dictionaries used in the data_fetch, data_clean, and feature_engineering scripts.

No functions are present, and no libraries or modules are required.
"""

season_team_to_coach_team_dict = {
     'BYU': 'Brigham Young',
     'Long Beach State': 'Cal State Long Beach',
     'Central Connecticut': 'Central Connecticut State',
     'UCF': 'Central Florida',
     'UConn': 'Connecticut',
     'Detroit': 'Detroit Mercy',
     'ETSU': 'East Tennessee State',
     'UIC': 'Illinois-Chicago',
     'LIU': 'Long Island University',
     'LSU': 'Louisiana State',
     'UMBC': 'Maryland-Baltimore County',
     'UMass': 'Massachusetts',
     'UMass-Lowell': 'Massachusetts-Lowell',
     'Ole Miss': 'Mississippi',
     'UMKC': 'Missouri-Kansas City',
     'UNLV': 'Nevada-Las Vegas',
     'UNC': 'North Carolina',
     'NC State': 'North Carolina State',
     'UNC Asheville': 'North Carolina-Asheville',
     # 'UNC Greensboro': 'North Carolina-Greensboro',
     'UNC Wilmington': 'North Carolina-Wilmington',
     'Penn': 'Pennsylvania',
     'Pitt': 'Pittsburgh',
     'SIU-Edwardsville': 'SIU Edwardsville',
     "St. Joseph's": "Saint Joseph's",
     "Saint Mary's": "Saint Mary's (CA)",
     "St. Peter's": "Saint Peter's",
     'USC Upstate': 'South Carolina Upstate',
     'USC': 'Southern California',
     'SMU': 'Southern Methodist',
     'Southern Miss': 'Southern Mississippi',
     'UT-Martin': 'Tennessee-Martin',
     'TCU': 'Texas Christian',
     'UTEP': 'Texas-El Paso',
     'UTSA': 'Texas-San Antonio',
     'UCSB': 'UC Santa Barbara',
     'California': 'University of California',
     'VCU': 'Virginia Commonwealth',
}

coach_team_to_mm_team_dict = {
     'UAB': 'Alabama-Birmingham',
     # 'UT Arlington': 'Texas-Arlington',
}

curr_season_to_tourney_dict = {
     # "Connecticut": "UConn",
     # 'Miami FL': 'Miami (FL)',
     # "Louisiana Lafayete": "Louisiana",
     # 'Brigham Young': 'BYU',
     # 'Louisiana State': 'LSU',
     # 'Loyola (IL)': 'Loyola Chicago',
     # 'Norfolk State': 'Norfolk St',
     # 'UCSB': 'UC Santa Barbara',
     # 'North Carolina-Greensboro': 'UNC Greensboro',
     # 'Southern California': 'USC',
     # 'Virginia Commonwealth': 'VCU',
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

# def hist_team_name_integrity_check(start_year, curr_year):
#     hist_stats_df, hist_coach_ranks_df, hist_games_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

#     for year in range(start_year, curr_year):
#         stats = fetch.get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/men/{year}-school-stats.html",
#                                      attrs={'id': 'basic_school_stats'})
#         hist_stats_df = pd.concat([hist_stats_df, stats], ignore_index=True)        

#         coaches = fetch.get_coach_rankings_data(year)
#         hist_coach_ranks_df = pd.concat([hist_coach_ranks_df, coaches], ignore_index=True)

#         mm_games = fetch.get_hist_bracket(year)
#         hist_games_df = pd.concat([hist_games_df, mm_games], ignore_index=True)


#     clean_hist_stats_df = clean_basic_stats(hist_stats_df)
#     clean_hist_stats_df['School'] = clean_hist_stats_df['School'].apply(lambda school: school[:-5])
#     clean_hist_stats_df['School'].replace(season_team_to_coach_team_dict, inplace=True)
#     school_stats_set = set(clean_hist_stats_df['School'])
    
#     hist_coach_ranks_df = pd.merge(clean_hist_stats_df, hist_coach_ranks_df,
#                             left_on='School', right_on='Coach_Team').drop('School', axis=1)
#     hist_coach_ranks_df['Coach_Team'].replace(coach_team_to_mm_team_dict, inplace=True)
#     school_coach_set = set(hist_coach_ranks_df['Coach_Team'])
    
#     school_games_set = set(hist_games_df['Team']).union(set(hist_games_df['Team.1']))

#     stats_coach_diff = school_stats_set.difference(school_coach_set)
#     coach_games_diff = school_coach_set.difference(school_games_set)

#     return stats_coach_diff, coach_games_diff

"""-------------------------------------------------------------------------------------------"""

# def curr_team_name_integrity_check(curr_year, curr_bracket_df):   
#     hist_coach_ranks_df = fetch.get_coach_rankings_data(curr_year)
    
#     school_coach_set = set(hist_coach_ranks_df['Coach_Team'])
#     school_games_set = set(curr_bracket_df['Team']).union(set(curr_bracket_df['Team.1']))
    
#     games_coach_diff = school_games_set.difference(school_coach_set)
    
#     return games_coach_diff
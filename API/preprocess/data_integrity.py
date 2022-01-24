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
     'UC-Santa Barbara': 'UCSB',
     'University of California': 'California',
     'Virginia Commonwealth': 'VCU',
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

# from data_clean import clean_merged_season_stats
# from data_pipeline import all_team_season_data

# def team_name_integrity_check(start_year, curr_year):
#     hist_stats_df, hist_coach_df = pd.DataFrame(), pd.DataFrame()

#     for year in range(start_year, curr_year):
#         coaches = fetch.get_coach_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-coaches.html")
#         hist_coach_df = pd.concat([hist_coach_df, coaches], ignore_index=True)

#         all_curr_season_data, curr_season_basic_df = all_team_season_data(year)
#         stats = clean_merged_season_stats(year, all_curr_season_data)
#         hist_stats_df = pd.concat([hist_stats_df, stats], ignore_index=True)

#     school_stats_set = set(hist_stats_df['School'])
#     school_coach_set = set(hist_coach_df['Coach_Team'])

#     stat_coach_teams_diff_pre = school_stats_set.difference(school_coach_set)

#     hist_stats_df['School'].replace(season_team_to_coach_team_dict, inplace=True)
#     school_stats_set = set(hist_stats_df['School'])

#     stat_coach_teams_diff_post = school_stats_set.difference(school_coach_set)

#     true_diff = stat_coach_teams_diff_post.difference(stat_coach_teams_diff_pre)
#     return true_diff
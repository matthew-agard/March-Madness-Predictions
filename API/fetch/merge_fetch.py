import pandas as pd
import numpy as np

ratings_team_to_coach_team_dict = {
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

playin_regions_list = [
     'east', 
     'west', 
     'south', 
     'southeast', 
     'southwest', 
     'midwest', 
     'albuquerque', 
     'atlanta', 
     'austin', 
     'chicago', 
     'eastrutherford',
     'minneapolis', 
     'oakland', 
     'phoenix', 
     'stlouis', 
     'syracuse', 
     'washington',
]

def merge_raw_tourney_games(year, seeds_list, teams_scores_list, rounds_list):
    # Convert play-in data lists into reshaped arrays so each team's data can be retrieved using the same x-axis index
    seeds_arr = np.array(seeds_list).reshape(len(seeds_list), 1)
    teams_scores_arr = np.array(teams_scores_list).reshape(len(teams_scores_list) // 2, 2)
    rounds_arr = np.array(rounds_list).reshape(len(rounds_list), 1)

    # Concatenate team arrays along y-axis (matching index)
    all_teams_data = np.concatenate((seeds_arr, teams_scores_arr), axis=1)
    # Convert array of teams into matchups
    teams_matchups = all_teams_data.reshape(len(teams_scores_list) // 4, 6)
    # Concatenate teams_matchups with round data arrays along y-axis (matching index)
    all_arr_data = np.concatenate((rounds_arr, teams_matchups), axis=1)
    
    # Populate array into a DataFrame
    games_df = pd.DataFrame(
        data=all_arr_data, 
        index=range(len(all_arr_data)), 
        columns=['Round', 'Seed', 'Team', 'Score', 'Seed.1', 'Team.1', 'Score.1']
    )
    
    # Add data for the calendar year (used alongside round name in EDA)
    games_df.insert(0, 'Year', len(games_df) * [year])
    
    # Format DataFrame to seamlessly integrate into custom API pipeline
    for col in games_df.columns:
        if any([val in col for val in ['Seed', 'Score', 'Year']]):
            games_df[col] = games_df[col].astype(int)

    return games_df
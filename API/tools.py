path.append('../API/preprocess')
from data_clean import clean_merged_season_stats
from data_pipeline import all_team_season_data

hist_brackets_df, hist_stats_df, hist_coach_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for year in range(start_year, curr_year):
    bracket = fetch.get_hist_bracket(url=f'https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html', year=year)
    hist_brackets_df = pd.concat([hist_brackets_df, bracket], ignore_index=True)
    
    coaches = fetch.get_coach_data(url=f"https://www.sports-reference.com/cbb/seasons/{year}-coaches.html")
    hist_coach_df = pd.concat([hist_coach_df, coaches], ignore_index=True)
    
    all_curr_season_data, curr_season_basic_df = all_team_season_data(year)
    stats = clean_merged_season_stats(year, all_curr_season_data, curr_season_basic_df)
    hist_stats_df = pd.concat([hist_stats_df, stats], ignore_index=True)

#--------------------------------------------------#

hist_stats_df['School'].replace(hist_season_to_tourney_dict, inplace=True)

school_stats_set = set(hist_stats_df['School'])
school_coach_set = set(hist_coach_df['Coach_Team'])
school_brackets_set = set(hist_brackets_df['Team']).union(set(hist_brackets_df['Team.1']))

diff = school_stats_set.difference(school_brackets_set)
diff2 = school_coach_set.difference(school_brackets_set)

# diff.difference(set(hist_season_to_tourney_dict.keys()))
diff

# set({5, 7, 12, 19, 2}).difference(set({2, 20, 15, 7, 12}))
import math
import sleeper_wrapper
from sleeper_wrapper import Stats
from requests import get, post
import pandas as pd
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

def get_all_team_rosters(lg, week):
    # Get the number of teams in the league
    num_teams = len(lg.teams())

    # Initialize an empty DataFrame to hold all rosters
    all_rosters_df = pd.DataFrame()

    # Iterate through each team
    for num in range(1, num_teams + 1):
        # Construct the team key
        teamkey = f'{lg.league_id}.t.{num}'

        # Get the roster for the team
        roster = lg.to_team(teamkey).roster(week = week)
        # Convert the roster to a DataFrame
        roster_df = pd.DataFrame(roster)

        # Add team number and owner name to the DataFrame
        roster_df['team_id'] = lg.teams()[teamkey]['team_id']
        roster_df['team_name'] = lg.teams()[teamkey]['name']

        # Append the team's roster DataFrame to the all_rosters DataFrame
        all_rosters_df = pd.concat([all_rosters_df, roster_df], ignore_index=True)
        all_rosters_df['player_id'] = all_rosters_df['player_id'].astype('str')

    return all_rosters_df

def get_stats_for_weeks(start_week, end_week, season_type, year, stat_type = 'projections'):
    # Start week is current week
    # Initialize an empty list to store DataFrames
    # gets all NFL players in the Sleeper system
    # players = sleeper_wrapper.Players()
    # all_players = players.get_all_players()
    # players_df = pd.DataFrame.from_dict(all_players, orient='index')
    # players_df['yahoo_id'] = players_df['yahoo_id'].fillna(0).astype(int).astype('str')
    # Loop through each week from start_week to end_week
    if stat_type == 'projections':
        dfs = []
        for week in range(start_week, end_week + 1):
            # Retrieve the projections for the given week
            week_projections = stats.get_week_projections(season_type, year, week)

            # Convert the projections to a DataFrame
            df_week = pd.DataFrame.from_dict(week_projections, orient='index')

            df_week.index.name = 'player_id'
            df_week.reset_index(inplace=True)

            # Add a column to indicate the week
            df_week['week'] = week

            # Append the DataFrame to the list
            dfs.append(df_week)
    elif stat_type == 'stats':
        dfs = []
        for week in range(1, start_week):
            # Retrieve the projections for the given week
            week_stats = stats.get_week_stats(season_type, year, week)

            # Convert the projections to a DataFrame
            df_week = pd.DataFrame.from_dict(week_stats, orient='index')

            df_week.index.name = 'player_id'
            df_week.reset_index(inplace=True)

            # Add a column to indicate the week
            df_week['week'] = week

            # Append the DataFrame to the list
            dfs.append(df_week)

    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, axis=0)

    # Reset the index of the combined DataFrame
    combined_df.reset_index(drop=True, inplace=True)
    combined_df=combined_df.rename(columns={'player_id':'sleeper_id'})
    # combined_df=combined_df.merge(players_df[['player_id', 'full_name']], on = 'player_id', how = 'inner')
    return combined_df

def calculate_top_n_averages(df, scoring_type, positions_of_interest, n=5):
    # Define the positions to calculate averages for
    position_averages = {}
    for position in positions_of_interest:
        # Filter the DataFrame for the specific position
        filtered_df = df[df['eligible_positions'].apply(lambda x: position in x)]

        # Custom function to calculate the mean of top n players per week
        def top_n_mean(group):
            sorted_group = group.sort_values(ascending=False)
            return sorted_group.head(n).mean()

        # Group by week and apply the custom function
        averages = filtered_df.groupby('week')[scoring_type].apply(top_n_mean)

        position_averages[position] = averages

    return position_averages

def calculate_dist_parameters(df, scoring_type):
    # Grouping by player_id and calculating mean and variance
    grouped_stats = df.groupby('sleeper_id')[scoring_type].agg(['mean', 'std']).reset_index()

    return grouped_stats[['sleeper_id', 'mean', 'std']]

def join_projections_and_stats(projections_df, stats_df, scoring_type = 'pts_half_ppr'):
    # scoring type = 'pts_std', 'pts_half_ppr', or 'pts_std'
    stats_subset = stats_df[['sleeper_id', 'week', 'pts_std', 'pts_half_ppr', 'pts_ppr']][~stats_df['pts_std'].isna() & ~stats_df['sleeper_id'].str.startswith('TEAM_')]
    results_df = calculate_dist_parameters(stats_subset, scoring_type)
    merged = projections_df.merge(results_df, on = 'sleeper_id', how = 'left')
    return merged

def set_optimal_lineup(df, roster_specs, scoring_system):
    # Initialize a column for optimal_position
    df['optimal_position'] = 'Bench'  # Default to bench

    # Iterate over each team and week
    for (team, week) in df.groupby(['team_name', 'week']).groups.keys():
        team_week_df = df[(df['team_name'] == team) & (df['week'] == week)].copy()

        # Initialize a dictionary to track positions needed
        positions_needed = {pos: specs['count'] for pos, specs in roster_specs.items() if specs['is_starting_position']}

        # First, assign players to unique positions
        for pos in positions_needed.keys():
            eligible_players = team_week_df[team_week_df['eligible_positions'].apply(lambda x: x == [pos])]
            for idx, player in eligible_players.iterrows():
                if positions_needed[pos] > 0:
                    team_week_df.at[idx, 'optimal_position'] = pos
                    positions_needed[pos] -= 1

        # Now assign players to shared positions based on highest points
        for pos, count in positions_needed.items():
            while count > 0:
                eligible_players = team_week_df[(team_week_df['optimal_position'] == 'Bench') & 
                                           (team_week_df['eligible_positions'].apply(lambda x: pos in x))]
                if not eligible_players.empty:
                    highest_scoring_player = eligible_players.sort_values(by=scoring_system, ascending=False).iloc[0]
                    team_week_df.at[highest_scoring_player.name, 'optimal_position'] = pos
                    count -= 1
                else:
                    break

        # Update the main dataframe
        df.update(team_week_df)
    return df

def find_missing_roster_spots(week_df, team_id):
    # Define the required positions in the lineup
    required_positions = ['QB', 'RB', 'WR', 'TE', 'W/R/T', 'DEF']
    
    # Filter the DataFrame for the specified team and week
    team_df = week_df[(week_df['team_id'] == team_id) & (week_df['week'] == week)]
    
    # Find which required positions are missing
    starting_positions = team_df['optimal_position'].tolist()
    missing_positions = [pos for pos in required_positions if pos not in starting_positions]

    return missing_positions

def simulate_week_points(week_df, week, imputable_evs, imputable_stds, scoring_type='pts_std', n=3, required_positions='NA'):
    # required_positions is a dict with key = position, value = number of starting positions

    week_df = week_df.copy()
    week_df = week_df.loc[(week_df['week'] == week) & (week_df['optimal_position'] != 'Bench')]

    # Simulate points for each player
    week_df['simulated_points'] = np.random.normal(week_df[scoring_type], week_df['std'])

    # Handle NaN values
    week_df['simulated_points'] = week_df['simulated_points'].fillna(0)

    # Initialize missing_data DataFrame
    missing_data = []

    # Find missing positions for all teams
    all_teams = week_df['team_id'].unique()
    for team_id in all_teams:
        team_data = week_df[week_df['team_id'] == team_id]
        
        # Check for each required position
        for pos, required_count in required_positions.items():
            # Filter team data for this position and check the count and NaNs
            na_missing = team_data[(team_data['optimal_position'] == pos)][scoring_type].isna().sum()
            empty_missing = team_data[(team_data['optimal_position'] == pos)][scoring_type].empty
            
            # Check if there are missing spots or NaN points for this position
            missing_count = na_missing + empty_missing
            if missing_count > 0:
                # Get imputable EV and STD for the position and week
                ev = imputable_evs[pos][week]
                std = imputable_stds[pos]
                
                # Generate simulated points for each missing spot
                for _ in range(missing_count):
                    simulated_points = max(np.random.normal(ev, std), 0)
                    missing_data.append({'team_id': team_id, 'total_points': simulated_points})

    missing_data = pd.DataFrame(missing_data)
    # Group by team and sum points
    team_points = week_df.groupby(['team_id', 'team_name'])['simulated_points'].sum().reset_index()
    team_points.columns = ['team_id', 'team_name', 'total_points']

    # Merge and update total points if there are missing data
    if not missing_data.empty:
        missing_data_agg = missing_data.groupby('team_id')['total_points'].sum().reset_index()
        team_points = pd.merge(team_points, missing_data_agg, on='team_id', how='left')
        team_points = team_points.fillna(0)
        team_points['total_points'] = team_points['total_points_x'] + team_points['total_points_y']
        team_points = team_points.drop(['total_points_x', 'total_points_y'], axis = 1)

    return team_points

def simulate_week_matchups(week_df, week, scoring_type, standings_df, rest_of_matchups_dict,
                           imputable_evs, imputable_stds, n, required_positions):
    matchups_dict = rest_of_matchups_dict[str(week)]
    standings_df = standings_df.copy(deep=True)
    team_points = simulate_week_points(week_df, week, imputable_evs, imputable_stds,
                                   scoring_type, n, required_positions='NA')

    for i in range(6):  # Assuming there are 6 matchups
        matchup_key = str(i)
        tm1 = matchups_dict['fantasy_content']['league'][1]['scoreboard']['0']['matchups'][matchup_key]['matchup']['0']['teams']['0']['team'][0][1]['team_id']
        tm2 = matchups_dict['fantasy_content']['league'][1]['scoreboard']['0']['matchups'][matchup_key]['matchup']['0']['teams']['1']['team'][0][1]['team_id']

        # Get points for each team in the matchup
        tm1_points = team_points.loc[team_points['team_id'] == tm1, 'total_points'].values[0]
        tm2_points = team_points.loc[team_points['team_id'] == tm2, 'total_points'].values[0]

        # Compare points and update standings
        if tm1_points > tm2_points:
            standings_df.loc[standings_df['team_id'] == tm1, 'wins'] += 1
            standings_df.loc[standings_df['team_id'] == tm2, 'losses'] += 1
            if standings_df.loc[standings_df['team_id'] == tm1, 'type'].values[0] == 'win':
                standings_df.loc[standings_df['team_id'] == tm1, 'value'] += 1
            else:
                standings_df.loc[standings_df['team_id'] == tm1, 'type'] = 'win'
                standings_df.loc[standings_df['team_id'] == tm1, 'value'] = 1
            if standings_df.loc[standings_df['team_id'] == tm2, 'type'].values[0] == 'loss':
                standings_df.loc[standings_df['team_id'] == tm2, 'value'] += 1
            else:
                standings_df.loc[standings_df['team_id'] == tm2, 'type'] = 'loss'
                standings_df.loc[standings_df['team_id'] == tm2, 'value'] = 1
        elif tm2_points > tm1_points:
            standings_df.loc[standings_df['team_id'] == tm2, 'wins'] += 1
            standings_df.loc[standings_df['team_id'] == tm1, 'losses'] += 1
            if standings_df.loc[standings_df['team_id'] == tm2, 'type'].values[0] == 'win':
                standings_df.loc[standings_df['team_id'] == tm2, 'value'] += 1
            else:
                standings_df.loc[standings_df['team_id'] == tm2, 'type'] = 'win'
                standings_df.loc[standings_df['team_id'] == tm2, 'value'] = 1
            if standings_df.loc[standings_df['team_id'] == tm1, 'type'].values[0] == 'loss':
                standings_df.loc[standings_df['team_id'] == tm1, 'value'] += 1
            else:
                standings_df.loc[standings_df['team_id'] == tm1, 'type'] = 'loss'
                standings_df.loc[standings_df['team_id'] == tm1, 'value'] = 1
        else:  # In case of a tie
            standings_df.loc[standings_df['team_id'] == tm1, 'ties'] += 1
            standings_df.loc[standings_df['team_id'] == tm2, 'ties'] += 1
            if standings_df.loc[standings_df['team_id'] == tm1, 'type'].values[0] == 'tie':
                standings_df.loc[standings_df['team_id'] == tm1, 'value'] += 1
            else:
                standings_df.loc[standings_df['team_id'] == tm1, 'type'] = 'tie'
                standings_df.loc[standings_df['team_id'] == tm1, 'value'] = 1
            if standings_df.loc[standings_df['team_id'] == tm2, 'type'].values[0] == 'tie':
                standings_df.loc[standings_df['team_id'] == tm2, 'value'] += 1
            else:
                standings_df.loc[standings_df['team_id'] == tm1, 'type'] = 'tie'
                standings_df.loc[standings_df['team_id'] == tm2, 'value'] = 1

        # Update points for and against
        standings_df.loc[standings_df['team_id'] == tm1, 'points_for'] += tm1_points
        standings_df.loc[standings_df['team_id'] == tm1, 'points_against'] += tm2_points
        standings_df.loc[standings_df['team_id'] == tm2, 'points_for'] += tm2_points
        standings_df.loc[standings_df['team_id'] == tm2, 'points_against'] += tm1_points
            
    # Update winning percentage and re-rank
    standings_df['percentage'] = (standings_df['wins'] + 0.5 * standings_df['ties']) / (standings_df['wins'] + standings_df['losses'] + standings_df['ties'])
    standings_df.sort_values(by=['percentage', 'points_for'], ascending=False, inplace=True)
    standings_df['rank'] = range(1, len(standings_df) + 1)
    standings_df['playoff_seed'] = range(1, len(standings_df) + 1)

    return standings_df

def simulate_end_of_regular_season(week_df, initial_standings_df, scoring_type, rest_of_matchups_dict, num_playoff_teams, imputable_evs,
                                           imputable_stds, required_positions='NA', n=3, week=14):
    current_standings_df = initial_standings_df.copy()
    for week in range(week, 15):  # Weeks 12, 13, 14
        matchups_dict = rest_of_matchups_dict[str(week)]  # Retrieve matchups for the current week
        updated_standings_df = simulate_week_matchups(week_df, week, scoring_type, standings_df, rest_of_matchups_dict, imputable_evs,
                                           imputable_stds, n=3, required_positions='NA')

        # Prepare for the next iteration (if any)
        current_standings_df = updated_standings_df
        # Update current_week_df with data for the next week
        # current_week_df = <Code to get data for the next week>
    current_standings_df['made_playoffs'] = np.where(current_standings_df['playoff_seed'] <= num_playoff_teams, 1,
                                                     0)
    return current_standings_df


def set_up_bracket(final_standings, num_playoff_teams):
    num_teams = len(final_standings)
    sorted_standings = final_standings.sort_values(by='rank')

    # Top teams for winners bracket, remaining for losers bracket
    winners = sorted_standings.head(num_playoff_teams)['team_id'].tolist()
    losers = sorted_standings.tail(num_teams - num_playoff_teams)['team_id'].tolist()

    # Create initial winners bracket matchups
    winners_bracket = {
        'round_1': {
            'matchup_1': [winners[3], winners[4]],  # 4 vs 5
            'matchup_2': [winners[2], winners[5]],  # 3 vs 6
            'byes': [winners[0], winners[1]]       # 1 and 2
        },
        'round_2': {
            'matchup_1': [winners[0]],  # To be filled with winner of 4 vs 5 vs 1st seed
            'matchup_2': [winners[1]],  # To be filled with winner of 3 vs 6 vs 2nd seed
            'byes': []        # To be filled with losing teams from round 1
        },
        'round_3': {
            'championship': [],
            '3rd_place': [],   # To be filled with losing teams from round 2
            '5th_place': []    # Automatically filled with byes from round 2
        }
    }

    # Create initial losers bracket matchups
    losers_bracket = {
        'round_1': {
            'matchup_1': [losers[1], losers[2]],  # 8 vs 9
            'matchup_2': [losers[0], losers[3]],  # 7 vs 10
            'byes': [losers[5], losers[4]]      # 12 and 11
        },
        'round_2': {
            'matchup_1': [losers[5]],  # To be filled with loser of 8 vs 9 vs 12th seed
            'matchup_2': [losers[4]],  # To be filled with loser of 7 vs 10 vs 11th seed
            'byes': []        # To be filled with winning teams from round 1
        },
        'round_3': {
            'toilet_bowl': [], # To be filled with winning teams from round 2
            '9th_place': [],   # Automatically filled with byes from round 2
            '7th_place': []    # To be filled with losing teams from round 2
        }
    }
    final_placement = {i:[] for i in range(1, 13)}

    return {'winners': winners_bracket, 'losers': losers_bracket, 'final_placement':final_placement}


def generate_final_standings(brackets, simulated_points):
    # Create a dictionary to map team IDs to team names
    team_names = pd.Series(simulated_points['team_name'].values, index=simulated_points['team_id']).to_dict()

    final_standings = []

    # Iterate through final placements and assign standings
    for placement, teams in brackets['final_placement'].items():
        for team_id in teams:
            final_standings.append({
                'team_id': team_id,
                'team_name': team_names.get(team_id, 'Unknown Team'),
                'final_placement': int(placement)
            })

    # Convert to DataFrame and sort by final placement
    final_standings_df = pd.DataFrame(final_standings)
    final_standings_df = final_standings_df.sort_values(by='final_placement').reset_index(drop=True)
    final_standings_df['won_championship'] = np.where(final_standings_df['final_placement'] == 1, 1,
                                                      0)
    final_standings_df['is_toilet_loser'] = np.where(final_standings_df['final_placement'] == 12, 1,
                                                      0)
    return final_standings_df

def check_and_append(dict_entry, team):
    if team in dict_entry:
        1
    else:
        dict_entry.append(team)
        
def check_and_append(dict_entry, team):
    if team in dict_entry:
        1
    else:
        dict_entry.append(team)
def update_next_round(brackets, key, current_round, winner, loser, matchup_key):
    if current_round == 1:
        if key == 'winners':
            # Determine the next round matchup for the winner
            next_round_matchup = 'matchup_1' if matchup_key == 'matchup_1' else 'matchup_2'
            check_and_append(brackets[key]['round_2'][next_round_matchup], winner)
        if key == 'losers':
            # Determine the next round matchup for the winner
            next_round_matchup = 'matchup_1' if matchup_key == 'matchup_1' else 'matchup_2'
            check_and_append(brackets[key]['round_2'][next_round_matchup], loser) 
        # Update the loser to the next round's byes in winners bracket
        if key == 'winners':
            check_and_append(brackets[key]['round_2']['byes'], loser)
            check_and_append(brackets[key]['round_3']['5th_place'], loser)
        elif key == 'losers':
            check_and_append(brackets[key]['round_2']['byes'], winner)
            check_and_append(brackets[key]['round_3']['7th_place'], winner)

    elif current_round == 2:
        # Determine the next round's matchup for the winner
        if key == 'winners':
            check_and_append(brackets[key]['round_3']['championship'], winner)
            check_and_append(brackets[key]['round_3']['3rd_place'], loser)
        else:  # Losers bracket
            check_and_append(brackets[key]['round_3']['toilet_bowl'], loser)
            check_and_append(brackets[key]['round_3']['9th_place'], winner)
    elif current_round == 3:
        if key == 'winners':
            if matchup_key == 'championship':
                check_and_append(brackets['final_placement'][1], winner)
                check_and_append(brackets['final_placement'][2], loser)
            elif matchup_key == '3rd_place':
                check_and_append(brackets['final_placement'][3], winner)
                check_and_append(brackets['final_placement'][4], loser)
            elif matchup_key == '5th_place':
                check_and_append(brackets['final_placement'][5], winner)
                check_and_append(brackets['final_placement'][6], loser)
        if key == 'losers':
            if matchup_key == 'toilet_bowl':
                check_and_append(brackets['final_placement'][11], winner)
                check_and_append(brackets['final_placement'][12], loser)
            elif matchup_key == '7th_place':
                check_and_append(brackets['final_placement'][7], winner)
                check_and_append(brackets['final_placement'][8], loser)
            elif matchup_key == '9th_place':
                check_and_append(brackets['final_placement'][9], winner)
                check_and_append(brackets['final_placement'][10], loser)

def simulate_playoffs_12(week_df, start_week, scoring_type, final_standings, num_playoff_teams, imputable_evs, imputable_stds,
                                n, required_positions):
    brackets = set_up_bracket(final_standings, num_playoff_teams=num_playoff_teams)

    for week in range(start_week, start_week + 3):  # Weeks 15, 16, 17 for playoffs
        # Simulate points for the current week
        simulated_points = simulate_week_points(week_df, week, imputable_evs, imputable_stds,
                                   scoring_type, n, required_positions='NA')

        # Determine the current round based on the week
        current_round = week - start_week + 1

        # Process matchups for the current round
        for key in ['winners', 'losers']:
            round_key = f'round_{current_round}'
            if round_key in brackets[key]:
                for matchup_key, teams in brackets[key][round_key].items():
                    if matchup_key != 'byes' and len(teams) == 2:  # Actual matchup
                        team1_id, team2_id = teams
                        team1_points = simulated_points[simulated_points['team_id'] == team1_id]['total_points'].values[0]
                        team2_points = simulated_points[simulated_points['team_id'] == team2_id]['total_points'].values[0]

                        winner = team1_id if team1_points > team2_points else team2_id
                        loser = team2_id if winner == team1_id else team1_id

                        # Update next round's matchups based on current round's winner
                        update_next_round(brackets, key, current_round, winner, loser, matchup_key)
    final_standings_df = generate_final_standings(brackets, simulated_points)
    return brackets, final_standings_df

def update_regular_season_stats(current_stats, new_data):
    for index, row in new_data.iterrows():
        team_id = row['team_id']
        if team_id not in current_stats:
            # If the team is not already in the stats, add it
            current_stats[team_id] = row
            current_stats[team_id]['best_playoff_seed'] = row['playoff_seed']
            current_stats[team_id]['worst_playoff_seed'] = row['playoff_seed']
            current_stats[team_id]['simulations_count'] = 1  # Initialize simulations count
        else:
            # Update existing stats with weighted averages and best/worst records
            for col in ['playoff_seed', 'points_for', 'points_against', 'wins', 'losses', 'ties', 'percentage']:
                current_stats[team_id][col] = (current_stats[team_id][col] * current_stats[team_id]['simulations_count'] + row[col]) / (current_stats[team_id]['simulations_count'] + 1)
            
            current_stats[team_id]['made_playoffs'] = (current_stats[team_id]['made_playoffs'] * current_stats[team_id]['simulations_count'] + row['made_playoffs']) / (current_stats[team_id]['simulations_count'] + 1)
            current_stats[team_id]['best_playoff_seed'] = min(current_stats[team_id]['best_playoff_seed'], row['playoff_seed'])
            current_stats[team_id]['worst_playoff_seed'] = max(current_stats[team_id]['worst_playoff_seed'], row['playoff_seed'])
            current_stats[team_id]['simulations_count'] += 1  # Increment simulations count
    return current_stats

def update_playoff_stats(current_stats, new_data):
    for index, row in new_data.iterrows():
        team_id = row['team_id']
        if team_id not in current_stats:
            # If the team is not already in the stats, add it
            current_stats[team_id] = row
            current_stats[team_id]['best_finish'] = row['final_placement']
            current_stats[team_id]['worst_finish'] = row['final_placement']
            current_stats[team_id]['simulations_count'] = 1  # Initialize simulations count
        else:
            # Update existing stats with weighted averages and best/worst finishes
            current_stats[team_id]['final_placement'] = (current_stats[team_id]['final_placement'] * current_stats[team_id]['simulations_count'] + row['final_placement']) / (current_stats[team_id]['simulations_count'] + 1)
            current_stats[team_id]['won_championship'] = (current_stats[team_id]['won_championship'] * current_stats[team_id]['simulations_count'] + row['won_championship']) / (current_stats[team_id]['simulations_count'] + 1)
            current_stats[team_id]['is_toilet_loser'] = (current_stats[team_id]['is_toilet_loser'] * current_stats[team_id]['simulations_count'] + row['is_toilet_loser']) / (current_stats[team_id]['simulations_count'] + 1)
            current_stats[team_id]['best_finish'] = min(current_stats[team_id]['best_finish'], row['final_placement'])
            current_stats[team_id]['worst_finish'] = max(current_stats[team_id]['worst_finish'], row['final_placement'])
            current_stats[team_id]['simulations_count'] += 1  # Increment simulations count
    return current_stats


def run_simulations(week_df, n, scoring_type, standings_df, rest_of_matchups_dict):
    # Initialize dictionaries to store cumulative stats
    regular_season_stats = {}
    playoff_stats = {}
    for _ in range(n):
        # Simulate the end of the regular season
        final_regular_season_standings = simulate_end_of_regular_season(week_df, standings_df, scoring_type, rest_of_matchups_dict, num_playoff_teams, imputable_evs,
                                           imputable_stds, n=3, required_positions='NA')
        # Update regular season stats with the latest simulation results
        regular_season_stats = update_regular_season_stats(regular_season_stats, final_regular_season_standings)

        # Simulate playoffs
        final_brackets, final_standings_playoffs = simulate_playoffs_12(week_df, 15, scoring_type=scoring_type, final_standings=final_regular_season_standings,
                                                                 imputable_evs=imputable_evs, imputable_stds=imputable_stds, required_positions='NA',n=3, num_playoff_teams=6)
        # Update playoff stats with the latest simulation results
        playoff_stats = update_playoff_stats(playoff_stats, final_standings_playoffs)

    # Convert the final stats dictionaries to DataFrames for easier analysis and export
    regular_season_stats_df = pd.DataFrame.from_dict(regular_season_stats, orient='index')
    cols_to_select2 = ['team_id', 'name', 'playoff_seed', 'points_for', 'points_against', 'wins', 'losses',
                       'ties', 'percentage', 'made_playoffs', 'best_playoff_seed', 'worst_playoff_seed']
    regular_season_stats_df = regular_season_stats_df[cols_to_select2]
    regular_season_stats_df = regular_season_stats_df.sort_values(by=['percentage', 'points_for'], ascending=False)
    playoff_stats_df = pd.DataFrame.from_dict(playoff_stats, orient='index')

    return regular_season_stats_df, playoff_stats_df
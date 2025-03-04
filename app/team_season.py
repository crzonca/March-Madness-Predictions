import networkx as nx
import pandas as pd
import numpy as np

from app import team_info


def get_games_graph(config, games_df=None):
    name_map = team_info.map_full_names_to_school(config)
    if games_df is None:
        games_df = pd.read_csv(config.get('resource_locations').get('stat_lines'))
        games_df = games_df.drop_duplicates(subset=['GameID', 'Team', 'TeamID', 'Opponent', 'OpponentID'])
    games_df['Team'] = games_df.apply(lambda r: name_map.get(r['Team'], r['Team']), axis=1)
    games_df['Opponent'] = games_df.apply(lambda r: name_map.get(r['Opponent'], r['Opponent']), axis=1)

    graph = nx.MultiDiGraph()

    for game_id in games_df['GameID'].unique():
        observations = games_df.loc[games_df['GameID'] == game_id]
        observations = observations.sort_values(by='PTS', ascending=False)

        if len(observations) < 2:
            observation = observations.iloc[0]
            if 'ND1' == observation['Opponent']:
                win = observation['WinOrLoss'] == 'W'
                if win:
                    graph.add_edge(observation['Opponent'], observation['Team'])
                else:
                    graph.add_edge(observation['Team'], observation['Opponent'])
            continue

        winner, loser = observations['Team']
        graph.add_edge(loser, winner)

    return graph


def get_location_games_graph(config, games_df=None):
    name_map = team_info.map_full_names_to_school(config)
    if games_df is None:
        games_df = pd.read_csv(config.get('resource_locations').get('stat_lines'))
        games_df = games_df.drop_duplicates(subset=['GameID', 'Team', 'TeamID', 'Opponent', 'OpponentID'])
    games_df['Team'] = games_df.apply(lambda r: name_map.get(r['Team'], r['Team']), axis=1)
    games_df['Opponent'] = games_df.apply(lambda r: name_map.get(r['Opponent'], r['Opponent']), axis=1)

    home_winner_graph = nx.MultiDiGraph()
    neut_winner_graph = nx.MultiDiGraph()
    away_winner_graph = nx.MultiDiGraph()

    for game_id in games_df['GameID'].unique():
        observations = games_df.loc[games_df['GameID'] == game_id]
        observations = observations.sort_values(by='PTS', ascending=False)

        if len(observations) < 2:
            observation = observations.iloc[0]
            if 'ND1' == observation['Opponent']:
                win = observation['WinOrLoss'] == 'W'
                location = observation['Location']
                if win:
                    if location == 'H':
                        home_winner_graph.add_edge(observation['Opponent'], observation['Team'])
                    elif location == 'V':
                        away_winner_graph.add_edge(observation['Opponent'], observation['Team'])
                    else:
                        neut_winner_graph.add_edge(observation['Opponent'], observation['Team'])
                else:
                    if location == 'H':
                        home_winner_graph.add_edge(observation['Team'], observation['Opponent'])
                    elif location == 'V':
                        away_winner_graph.add_edge(observation['Team'], observation['Opponent'])
                    else:
                        neut_winner_graph.add_edge(observation['Team'], observation['Opponent'])
            continue

        winner_loc, loser_loc = observations['Location']

        if len(observations) < 2:
            observation = observations.iloc[0]
            if 'ND1' == observation['Opponent']:
                win = observation['WinOrLoss'] == 'W'
                if win:
                    if winner_loc == 'H':
                        home_winner_graph.add_edge(observation['Opponent'], observation['Team'])
                    elif winner_loc == 'V':
                        away_winner_graph.add_edge(observation['Opponent'], observation['Team'])
                    else:
                        neut_winner_graph.add_edge(observation['Opponent'], observation['Team'])
                else:
                    if winner_loc == 'H':
                        home_winner_graph.add_edge(observation['Team'], observation['Opponent'])
                    elif winner_loc == 'V':
                        away_winner_graph.add_edge(observation['Team'], observation['Opponent'])
                    else:
                        neut_winner_graph.add_edge(observation['Team'], observation['Opponent'])
            continue

        winner, loser = observations['Team']

        if winner_loc == 'H':
            home_winner_graph.add_edge(loser, winner)
        elif winner_loc == 'V':
            away_winner_graph.add_edge(loser, winner)
        else:
            neut_winner_graph.add_edge(loser, winner)

    return home_winner_graph, neut_winner_graph, away_winner_graph

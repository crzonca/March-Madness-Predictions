import itertools
import math

import choix
import networkx as nx
import numpy as np
import pandas as pd

from app import tournament


# Model 1: Pure BT
def get_ranking_and_chance(bracket, games_graph, alpha=1, use_all=True):
    ranking_df = get_team_ranking(games_graph, alpha=alpha)
    chance_df = create_chance_df(bracket, ranking_df, use_all=use_all)
    return ranking_df, chance_df


def get_team_ranking(games_graph, alpha=1):
    nodes = games_graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(games_graph), columns=nodes)
    df.index = nodes

    teams = list(df.index)
    df = df.fillna(0)

    teams_to_index = {team: i for i, team in enumerate(teams)}
    index_to_teams = {i: team for team, i in teams_to_index.items()}

    graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    edges = [list(itertools.repeat((teams_to_index.get(team2),
                                    teams_to_index.get(team1)),
                                   int(weight_dict.get('weight'))))
             for team1, team2, weight_dict in graph.edges.data()]
    edges = list(itertools.chain.from_iterable(edges))

    try:
        coeffs, cov = choix.ep_pairwise(n_items=len(teams), data=edges, alpha=alpha)
        coeffs = pd.Series(coeffs)
        cov = pd.Series(cov.diagonal())
        coef_df = pd.DataFrame([coeffs, cov]).T
        coef_df.columns = ['BT', 'Var']
        coef_df.index = [index_to_teams.get(index) for index in coef_df.index]
    except np.linalg.LinAlgError:
        coeffs = pd.Series(choix.opt_pairwise(n_items=len(teams), data=edges))
        coeffs = coeffs.sort_values(ascending=False)
        coeffs = {index_to_teams.get(index): coeff for index, coeff in coeffs.items()}
        coef_df = pd.DataFrame(columns=['BT', 'Var'], index=coeffs.keys())
        for team, bt in coeffs.items():
            coef_df.at[team, 'BT'] = coeffs.get(team)
            coef_df.at[team, 'Var'] = 1.0
    coef_df = coef_df.sort_values(by='BT', ascending=False)
    return coef_df


def create_chance_df(bracket, bt_df, use_all=True):
    bt_dict = {team: row['BT'] for team, row in bt_df.iterrows()}

    teams = bt_dict.keys() if use_all else tournament.get_bracket_teams(bracket)
    chances_df = pd.DataFrame(index=teams, columns=teams)
    for team1, team2 in itertools.combinations(teams, 2):
        team1_bt = bt_dict.get(team1, 0)
        team2_bt = bt_dict.get(team2, 0)
        if team1 not in bt_dict:
            print(team1, 'MISSING')
            print('\n'.join(sorted(bt_dict.keys())))
            raise ValueError
        if team2 not in bt_dict:
            print(team2, 'MISSING')
            print('\n'.join(sorted(bt_dict.keys())))
            raise ValueError

        chance = math.exp(team1_bt) / (math.exp(team1_bt) + math.exp(team2_bt))
        chances_df.at[team2, team1] = chance
        chances_df.at[team1, team2] = 1 - chance

    return chances_df

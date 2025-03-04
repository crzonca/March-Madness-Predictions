import itertools
import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app import tournament


# Model 3: BT with Home Court Advantage and Road Disadvantage
def get_ranking_and_chance(bracket, graph, home_winner_graph, neutral_graph, away_winner_graph, alpha=1, use_all=True):
    ranking_df = get_team_ranking(graph, home_winner_graph, neutral_graph, away_winner_graph, alpha=alpha)
    chance_df = create_chance_df(bracket, ranking_df, use_all=use_all)
    return ranking_df, chance_df


def get_team_ranking(graph, home_winner_graph, neutral_graph, away_winner_graph, alpha=1):
    nodes = graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(graph), columns=nodes)
    df.index = nodes

    teams = list(df.index)

    teams_to_index = {team: i for i, team in enumerate(teams)}
    index_to_teams = {i: team for team, i in teams_to_index.items()}

    # -------------------------------

    nodes = home_winner_graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(home_winner_graph), columns=nodes)
    df.index = nodes
    df = df.fillna(0)

    home_winner_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    home_edges = [list(itertools.repeat((teams_to_index.get(team2),
                                         teams_to_index.get(team1)),
                                        int(weight_dict.get('weight'))))
                  for team1, team2, weight_dict in home_winner_graph.edges.data()]
    home_edges = list(itertools.chain.from_iterable(home_edges))

    # ---------------------------------

    nodes = away_winner_graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(away_winner_graph), columns=nodes)
    df.index = nodes
    df = df.fillna(0)

    away_winner_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    away_edges = [list(itertools.repeat((teams_to_index.get(team2),
                                         teams_to_index.get(team1)),
                                        int(weight_dict.get('weight'))))
                  for team1, team2, weight_dict in away_winner_graph.edges.data()]
    away_edges = list(itertools.chain.from_iterable(away_edges))

    # -------------------------------

    nodes = neutral_graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(neutral_graph), columns=nodes)
    df.index = nodes
    df = df.fillna(0)

    neutral_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    neutral_edges = [list(itertools.repeat((teams_to_index.get(team2),
                                            teams_to_index.get(team1)),
                                           int(weight_dict.get('weight'))))
                     for team1, team2, weight_dict in neutral_graph.edges.data()]
    neutral_edges = list(itertools.chain.from_iterable(neutral_edges))

    num_teams = len(teams)

    def safe_exp(x):
        x = min(x, 500)
        return math.exp(x)

    def objective(params):
        """Compute the negative penalized log-likelihood."""
        val = alpha * np.sum(params ** 2)
        for win, los in home_edges:  # Games where the home team won
            win_home = win + num_teams  # lambda i
            los_away = los + num_teams * 2  # theta  j
            val += np.logaddexp(0, -(params[win] + params[win_home] - params[los] - params[los_away]))
        for win, los in away_edges:  # Games where the away team won
            los_home = los + num_teams  # lambda j
            win_away = win + num_teams * 2  # theta  i
            val += np.logaddexp(0, -(params[win] + params[win_away] - params[los] - params[los_home]))
        for win, los in neutral_edges:
            val += np.logaddexp(0, -(params[win] - params[los]))
        return val

    def gradient(params):
        grad = 2 * alpha * params
        for win, los in home_edges:
            # func = log(1 + e^(-(x+a-y-b)))
            #   x is home strength          (beta i)
            #   y is away strength          (beta j)
            #   a is home advantage         (lambda i)
            #   b is roaw disadvantage      (theta j)
            win_home = win + num_teams
            los_away = los + num_teams * 2
            z = 1 / (1 + safe_exp(params[win] + params[win_home] - params[los] - params[los_away]))
            grad[win] += -z  # func d/dx
            grad[los] += +z  # func d/dy
            grad[win_home] += -z  # func d/da
            grad[los_away] += +z  # func d/db
        for win, los in away_edges:
            # func = log(1 + e^(-(x+c-y-d)))
            #   x is home strength
            #   y is away strength
            #   c is roaw disadvantage
            #   d is home advantage
            los_home = los + num_teams
            win_away = win + num_teams * 2
            z = 1 / (1 + safe_exp(params[win] + params[win_away] - params[los] - params[los_home]))
            grad[win] += -z  # func d/dx
            grad[los] += +z  # func d/dy
            grad[win_away] += -z  # func d/dc
            grad[los_home] += +z  # func d/dd
        for win, los in neutral_edges:
            z = 1 / (1 + safe_exp(params[win] - params[los]))
            grad[win] += -z
            grad[los] += +z
        return grad

    x0 = np.zeros(num_teams * 3)
    res = minimize(objective, x0, method='BFGS', jac=gradient)

    index_to_teams.update({index + num_teams: index_to_teams.get(index) + ' Home Court'
                           for index in index_to_teams.keys()})
    index_to_teams.update({index + num_teams * 2: index_to_teams.get(index) + ' Road Dis'
                           for index in index_to_teams.keys()})
    coeffs = pd.Series(res.x)
    coeffs = coeffs.sort_values(ascending=False)
    coeffs.index = [index_to_teams.get(index) for index in coeffs.index]

    coeff_df = pd.DataFrame(index=teams_to_index.keys(), columns=['BT', 'Home Court Adv', 'Road Disadv'])

    for team in teams_to_index.keys():
        coeff_df.at[team, 'BT'] = coeffs.loc[team]
        coeff_df.at[team, 'Home Court Adv'] = coeffs.loc[team + ' Home Court']
        coeff_df.at[team, 'Road Disadv'] = coeffs.loc[team + ' Road Dis']

    return coeff_df


def create_chance_df(bracket, bt_loc_df, use_all=True):
    teams = bt_loc_df.index if use_all else tournament.get_bracket_teams(bracket)
    home_chances_df = pd.DataFrame(index=teams, columns=teams)
    chances_df = pd.DataFrame(index=teams, columns=teams)
    away_chances_df = pd.DataFrame(index=teams, columns=teams)
    for team1, team2 in itertools.combinations(teams, 2):
        team1_bt = bt_loc_df.at[team1, 'BT']
        team2_bt = bt_loc_df.at[team2, 'BT']
        team1_hc = bt_loc_df.at[team1, 'Home Court Adv']
        team2_hc = bt_loc_df.at[team2, 'Home Court Adv']
        team1_rd = bt_loc_df.at[team1, 'Road Disadv']
        team2_rd = bt_loc_df.at[team2, 'Road Disadv']

        t1_home_chance = math.exp(team1_bt + team1_hc) / (math.exp(team1_bt + team1_hc) + math.exp(team2_bt + team2_rd))
        t1_away_chance = math.exp(team1_bt + team1_rd) / (math.exp(team1_bt + team1_rd) + math.exp(team2_bt + team2_hc))
        neut_chance = math.exp(team1_bt) / (math.exp(team1_bt) + math.exp(team2_bt))

        home_chances_df.at[team2, team1] = t1_home_chance
        home_chances_df.at[team1, team2] = 1 - t1_away_chance
        chances_df.at[team2, team1] = neut_chance
        chances_df.at[team1, team2] = 1 - neut_chance
        away_chances_df.at[team2, team1] = t1_away_chance
        away_chances_df.at[team1, team2] = 1 - t1_home_chance

    return home_chances_df, chances_df, away_chances_df

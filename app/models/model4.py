import itertools
import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app import team_info
from app import tournament


# Model 4: BT with Offset for Conference Strength
def get_ranking_and_chance(bracket, graph, config, alpha=1, use_all=True):
    ranking_df = get_team_ranking(graph, config, alpha=alpha)
    chance_df = create_chance_df(bracket, ranking_df, use_all=use_all)
    return ranking_df, chance_df


def get_team_ranking(graph, config, alpha=1):
    nodes = graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(graph), columns=nodes)
    df.index = nodes

    teams = list(df.index)

    teams_to_index = {team: i for i, team in enumerate(teams)}
    index_to_teams = {i: team for team, i in teams_to_index.items()}

    nodes = graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(graph), columns=nodes)
    df.index = nodes
    df = df.fillna(0)

    graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    edges = [list(itertools.repeat((teams_to_index.get(team2),
                                    teams_to_index.get(team1)),
                                   int(weight_dict.get('weight'))))
             for team1, team2, weight_dict in graph.edges.data()]
    edges = list(itertools.chain.from_iterable(edges))

    num_teams = len(teams)

    conference_mapping = team_info.conference_mapping(config)
    conferences = set(conference_mapping.values())
    conference_index_mapping = {conf: index for index, conf in enumerate(conferences)}
    conference_index_mapping['Independent'] = len(conferences)
    index_conference_mapping = {i: conf for conf, i in conference_index_mapping.items()}
    i = 0

    def safe_exp(x):
        x = min(x, 500)
        return math.exp(x)

    def objective(params):
        """Compute the negative penalized log-likelihood."""
        val = alpha * np.sum(params ** 2)
        for win, los in edges:  # Games where the home team won
            win_name = index_to_teams.get(win)
            win_conf = conference_mapping.get(win_name, 'Independent')
            win_conf_index = conference_index_mapping.get(win_conf)

            los_name = index_to_teams.get(los)
            los_conf = conference_mapping.get(los_name, 'Independent')
            los_conf_index = conference_index_mapping.get(los_conf)

            win_conf_index = num_teams + win_conf_index  # lambda i
            los_conf_index = num_teams + los_conf_index  # lambda i
            val += np.logaddexp(0, -(params[win] + params[win_conf_index] - params[los] - params[los_conf_index]))
        return val

    def gradient(params):
        grad = 2 * alpha * params
        for win, los in edges:
            win_name = index_to_teams.get(win)
            win_conf = conference_mapping.get(win_name, 'Independent')
            win_conf_index = conference_index_mapping.get(win_conf)

            los_name = index_to_teams.get(los)
            los_conf = conference_mapping.get(los_name, 'Independent')
            los_conf_index = conference_index_mapping.get(los_conf)

            win_conf_index = num_teams + win_conf_index  # lambda i
            los_conf_index = num_teams + los_conf_index  # lambda i

            z = 1 / (1 + safe_exp(params[win] + params[win_conf_index] - params[los] - params[los_conf_index]))
            grad[win] += -z  # func d/dx
            grad[los] += +z  # func d/dy
            grad[win_conf_index] += -z  # func d/da
            grad[los_conf_index] += +z  # func d/db
        return grad

    x0 = np.zeros(num_teams + len(conferences) + 1)
    res = minimize(objective, x0, method='BFGS', jac=gradient)

    index_to_teams.update(
        {index + num_teams: 'Conf ' + index_conference_mapping.get(index) for index in index_conference_mapping.keys()})
    coeffs = pd.Series(res.x)
    coeffs = coeffs.sort_values(ascending=False)
    coeffs.index = [index_to_teams.get(index) for index in coeffs.index]

    coeff_df = pd.DataFrame(index=teams_to_index.keys(), columns=['Team BT', 'Conference', 'Conference BT', 'BT'])

    for team in teams_to_index.keys():
        coeff_df.at[team, 'Team BT'] = coeffs.loc[team]
        conference = conference_mapping.get(team, 'Independent')
        coeff_df.at[team, 'Conference'] = conference
        coeff_df.at[team, 'Conference BT'] = coeffs.loc['Conf ' + conference]

    coeff_df['BT'] = coeff_df.apply(lambda r: r['Team BT'] + r['Conference BT'], axis=1)

    coeff_df = coeff_df.sort_values(by=['Conference BT', 'Team BT'], kind='mergesort', ascending=[False, False])
    return coeff_df


def create_chance_df(bracket, conf_bt_df, use_all=False):
    teams = conf_bt_df.index if use_all else tournament.get_bracket_teams(bracket)
    chances_df = pd.DataFrame(index=teams, columns=teams)
    for team1, team2 in itertools.combinations(teams, 2):
        team1_bt = conf_bt_df.at[team1, 'BT']
        team2_bt = conf_bt_df.at[team2, 'BT']

        neut_chance = math.exp(team1_bt) / (math.exp(team1_bt) + math.exp(team2_bt))

        chances_df.at[team2, team1] = neut_chance
        chances_df.at[team1, team2] = 1 - neut_chance

    return chances_df

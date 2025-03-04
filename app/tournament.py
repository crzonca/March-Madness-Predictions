from itertools import repeat

import networkx as nx
import numpy as np
import pandas as pd
import json


def update_tournament_chances(config, chances_df):
    rounds = ['Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Winner']

    with open(config.get('resource_locations').get('tournament_results'), 'r') as f:
        tournament_results = json.load(f)

    # First Round
    for tournament_round in rounds:
        for winner in tournament_results.get('Round of 64').get('Winners'):
            chances_df.at[tournament_round, winner] = (chances_df.at[tournament_round, winner] /
                                                       chances_df.at['Round of 32', winner])

        for loser in tournament_results.get('Round of 64').get('Losers'):
            chances_df.at[tournament_round, loser] = 0.0

    # Second Round
    for tournament_round in rounds[1:]:
        for winner in tournament_results.get('Round of 32').get('Winners'):
            chances_df.at[tournament_round, winner] = (chances_df.at[tournament_round, winner] /
                                                       chances_df.at['Sweet 16', winner])

        for loser in tournament_results.get('Round of 32').get('Losers'):
            chances_df.at[tournament_round, loser] = 0.0

    # Sweet 16
    for tournament_round in rounds[2:]:
        for winner in tournament_results.get('Sweet 16').get('Winners'):
            chances_df.at[tournament_round, winner] = (chances_df.at[tournament_round, winner] /
                                                       chances_df.at['Elite 8', winner])

        for loser in tournament_results.get('Sweet 16').get('Losers'):
            chances_df.at[tournament_round, loser] = 0.0

    # Elite 8
    for tournament_round in rounds[3:]:
        for winner in tournament_results.get('Elite 8').get('Winners'):
            chances_df.at[tournament_round, winner] = (chances_df.at[tournament_round, winner] /
                                                       chances_df.at['Final 4', winner])

        for loser in tournament_results.get('Elite 8').get('Losers'):
            chances_df.at[tournament_round, loser] = 0.0

    # Final 4
    for tournament_round in rounds[4:]:
        for winner in tournament_results.get('Final 4').get('Winners'):
            chances_df.at[tournament_round, winner] = (chances_df.at[tournament_round, winner] /
                                                       chances_df.at['Championship', winner])

        for loser in tournament_results.get('Final 4').get('Losers'):
            chances_df.at[tournament_round, loser] = 0.0

    # Championship
    for tournament_round in rounds[5:]:
        for winner in tournament_results.get('Championship').get('Winners'):
            chances_df.at[tournament_round, winner] = (chances_df.at[tournament_round, winner] /
                                                       chances_df.at['Winner', winner])

        for loser in tournament_results.get('Championship').get('Losers'):
            chances_df.at[tournament_round, loser] = 0.0

    points_df = chances_df.copy()
    points_df['Round of 32'] = points_df['Round of 32'] * 10
    points_df['Sweet 16'] = points_df['Sweet 16'] * 20
    points_df['Elite 8'] = points_df['Elite 8'] * 40
    points_df['Final 4'] = points_df['Final 4'] * 80
    points_df['Championship'] = points_df['Championship'] * 160
    points_df['Winner'] = points_df['Winner'] * 320
    points_df = points_df.drop(columns=['Round of 64'])
    points_df['Total'] = points_df.apply(lambda r: r['Round of 32'] +
                                                   r['Sweet 16'] +
                                                   r['Elite 8'] +
                                                   r['Final 4'] +
                                                   r['Championship'] +
                                                   r['Winner'], axis=1)
    points_df = points_df.sort_values(by='Total', kind='mergesort', ascending=False)

    return chances_df, points_df


def get_play_in_teams(bracket):
    play_in_teams = list()
    play_in_names = ['Play In ' + str(n) for n in range(1, 5)]
    for sub_bracket in play_in_names:
        sub_bracket = bracket.get_sub_bracket(sub_bracket)
        play_in_teams.append(sub_bracket.left.label)
        play_in_teams.append(sub_bracket.right.label)
    return play_in_teams


def get_bracket_teams(bracket):
    return [leaf.label for leaf in bracket.get_leaves()]


def calculate_overall_chances_points(bracket, chances_df):
    teams = [leaf.label for leaf in bracket.get_leaves()]

    rounds = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Winner']
    game_chances = pd.DataFrame(index=teams, columns=rounds)

    for team in teams:
        team_path = list(nx.all_simple_paths(bracket.to_graph(), team, 'Winner'))[0]
        for bracket_round, bracket_location in zip(rounds, team_path[-7:]):
            sub_bracket = bracket.get_sub_bracket(bracket_location)
            game_chances.at[team, bracket_round] = sub_bracket.get_victory_chance(team, chances_df)

    actual_total_chances = [game_chances[bracket_round].sum() for bracket_round in rounds]
    chance_validation = np.isclose(actual_total_chances, np.logspace(6, 0, 7, base=2))
    if not all(chance_validation):
        bad_rounds = [r for r, close in zip(rounds, chance_validation) if not close]
        for bad_round in bad_rounds:
            print('Incorrect calculation at the', bad_round)

    march_madness_chance_df = game_chances.sort_values(by=list(reversed(rounds)),
                                                       kind='mergesort',
                                                       ascending=list(repeat(False, 7)))

    march_madness_points_df = march_madness_chance_df.copy()
    march_madness_points_df['Round of 32'] = march_madness_points_df['Round of 32'] * 10
    march_madness_points_df['Sweet 16'] = march_madness_points_df['Sweet 16'] * 20
    march_madness_points_df['Elite 8'] = march_madness_points_df['Elite 8'] * 40
    march_madness_points_df['Final 4'] = march_madness_points_df['Final 4'] * 80
    march_madness_points_df['Championship'] = march_madness_points_df['Championship'] * 160
    march_madness_points_df['Winner'] = march_madness_points_df['Winner'] * 320
    march_madness_points_df = march_madness_points_df.drop(columns=['Round of 64'])
    march_madness_points_df['Total'] = march_madness_points_df.apply(lambda r: r['Round of 32'] +
                                                                               r['Sweet 16'] +
                                                                               r['Elite 8'] +
                                                                               r['Final 4'] +
                                                                               r['Championship'] +
                                                                               r['Winner'], axis=1)
    march_madness_points_df = march_madness_points_df.sort_values(by='Total', kind='mergesort', ascending=False)

    return march_madness_chance_df, march_madness_points_df


def calculate_conditional_chances_points(march_madness_chance_df):
    cond_chance64, cond_points64 = get_conditional_chances_points(march_madness_chance_df, 'Round of 64')
    cond_chance32, cond_points32 = get_conditional_chances_points(march_madness_chance_df, 'Round of 32')
    cond_chance16, cond_points16 = get_conditional_chances_points(march_madness_chance_df, 'Sweet 16')
    cond_chance8, cond_points8 = get_conditional_chances_points(march_madness_chance_df, 'Elite 8')
    cond_chance4, cond_points4 = get_conditional_chances_points(march_madness_chance_df, 'Final 4')
    cond_chance2, cond_points2 = get_conditional_chances_points(march_madness_chance_df, 'Championship')

    march_madness_conditional_chance_df = pd.DataFrame(index=march_madness_chance_df.index,
                                                       columns=march_madness_chance_df.columns)
    march_madness_conditional_chance_df['Round of 64'] = cond_chance64
    march_madness_conditional_chance_df['Round of 32'] = cond_chance32
    march_madness_conditional_chance_df['Sweet 16'] = cond_chance16
    march_madness_conditional_chance_df['Elite 8'] = cond_chance8
    march_madness_conditional_chance_df['Final 4'] = cond_chance4
    march_madness_conditional_chance_df['Championship'] = cond_chance2
    march_madness_conditional_chance_df = march_madness_conditional_chance_df.drop(columns=['Winner'])

    march_madness_conditional_points_df = pd.DataFrame(index=march_madness_chance_df.index,
                                                       columns=march_madness_chance_df.columns)
    march_madness_conditional_points_df['Round of 64'] = cond_points64
    march_madness_conditional_points_df['Round of 32'] = cond_points32
    march_madness_conditional_points_df['Sweet 16'] = cond_points16
    march_madness_conditional_points_df['Elite 8'] = cond_points8
    march_madness_conditional_points_df['Final 4'] = cond_points4
    march_madness_conditional_points_df['Championship'] = cond_points2
    march_madness_conditional_points_df = march_madness_conditional_points_df.drop(columns=['Winner'])

    return march_madness_conditional_chance_df, march_madness_conditional_points_df


def get_conditional_chances_points(march_madness_chance_df, bracket_round):
    rounds = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Winner']
    round_index = rounds.index(bracket_round)
    drop_columns = rounds[:round_index + 1]
    remaining_rounds = rounds[round_index + 1:]
    round_points_map = {'Round of 32': 10,
                        'Sweet 16': 20,
                        'Elite 8': 40,
                        'Final 4': 80,
                        'Championship': 160,
                        'Winner': 320}

    march_madness_cond_chance_df = march_madness_chance_df.copy()
    march_madness_cond_chance_df = march_madness_cond_chance_df.div(march_madness_cond_chance_df[bracket_round], axis=0)

    march_madness_cond_points_df = march_madness_cond_chance_df.copy()
    for remaining_round in remaining_rounds:
        march_madness_cond_points_df[remaining_round] = march_madness_cond_points_df[
                                                            remaining_round] * round_points_map.get(remaining_round, 0)
    march_madness_cond_points_df = march_madness_cond_points_df.drop(columns=drop_columns)
    march_madness_cond_points_df = march_madness_cond_points_df.cumsum(axis=1)

    return march_madness_cond_chance_df['Winner'], march_madness_cond_points_df['Winner']


def get_selection_loss(bracket, march_madness_points_df, team, bracket_round):
    possible_matchups = bracket.get_all_opponents(team)
    all_rounds = ['Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Winner']
    relevant_rounds = all_rounds[:all_rounds.index(bracket_round) + 1]
    possible_matchups = {br_round: possible_opponents for br_round, possible_opponents in possible_matchups.items()
                         if br_round in relevant_rounds}

    total_loss = 0
    for br_round, possible_opponents in possible_matchups.items():
        future_rounds = all_rounds[all_rounds.index(br_round):]
        br_round_loss = march_madness_points_df.loc[possible_opponents][future_rounds].values.sum()
        total_loss += br_round_loss

    return total_loss

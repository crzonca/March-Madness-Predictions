import json

import pandas as pd


def validate_data(config, graph):
    with open(config.get('resource_locations').get('true_records'), 'r') as f:
        true_records = pd.read_csv(f)

    data_valid = True
    for index, row in true_records.iterrows():
        team = row['team']
        true_wins = row['wins']
        true_losses = row['losses']

        if team not in graph.nodes:
            print('No games found for', team)
            data_valid = False
            print()

        data_wins = graph.in_degree(team)
        data_losses = graph.out_degree(team)

        if true_wins != data_wins or true_losses != data_losses:
            if true_wins != data_wins:
                print('The data shows', team, 'have', data_wins, 'wins when they should have', true_wins)
                print('\tData shows they defeated:')
                defeated_teams = [t[0] for t in list(graph.in_edges(team)) if not pd.isna(t[0])]
                for defeated_team in set(defeated_teams):
                    num_times = defeated_teams.count(defeated_team)
                    print('\t', defeated_team.ljust(25), 'x' + str(num_times))
                data_valid = False

            if true_losses != data_losses:
                print('The data shows', team, 'have', data_losses, 'losses when they should have',
                      true_losses)
                print('\tData shows they lost to:')
                lost_to_teams = [t[1] for t in list(graph.out_edges(team)) if not pd.isna(t[1])]
                for lost_to_team in set(lost_to_teams):
                    num_times = lost_to_teams.count(lost_to_team)
                    print('\t', lost_to_team.ljust(25), 'x' + str(num_times))
                data_valid = False
            print()

    return data_valid


def get_bracket_similarity(bracket_df1, bracket_df2):
    teams = bracket_df1.index

    team_win_diffs = dict()
    for team in teams:
        bracket1_team = bracket_df1[team]
        bracket2_team = bracket_df2[team]
        difference = abs(bracket1_team - bracket2_team)
        win_diff = sum(difference)
        team_win_diffs[team] = win_diff

    team_win_diffs = {team: wins for team, wins in sorted(team_win_diffs.items(), key=lambda t: t[1], reverse=True)}
    different_games = sum(team_win_diffs.values()) / 2
    return different_games

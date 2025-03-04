import itertools

import networkx as nx
import pandas as pd

from app import tournament


def complete_bracket(bracket, objective_df, topo_df, choice_df_func):
    choice_df = choice_df_func(bracket, objective_df)
    topology = create_topology(topo_df, choice_df)
    topology = topology_bracket(bracket, topology)
    return topology


def create_choice_df_better_team(bracket, ranking_df):
    # At each comparison, choose the team that is better
    print('\t\tBetter team topology')

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)

    ranking_criterion = 'BT' if 'BT' in ranking_df.columns else 'Completeness'

    for team1, team2 in itertools.combinations(teams, 2):
        team1_score = ranking_df.at[team1, ranking_criterion]
        team2_score = ranking_df.at[team2, ranking_criterion]
        choice_df.at[team2, team1] = 1 if team1_score > team2_score else 0
        choice_df.at[team1, team2] = 1 if team2_score > team1_score else 0

    return choice_df


def create_choice_df_overall_chance(bracket, march_madness_chance_df):
    # At each comparison, choose the team that has the greater overall chance to win the tournament
    print('\t\tOverall chance topology')

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)

    for team1, team2 in itertools.combinations(teams, 2):
        team1_total_chance = march_madness_chance_df.at[team1, 'Winner']
        team2_total_chance = march_madness_chance_df.at[team2, 'Winner']
        choice_df.at[team2, team1] = 1 if team1_total_chance > team2_total_chance else 0
        choice_df.at[team1, team2] = 1 if team2_total_chance > team1_total_chance else 0

    return choice_df


def create_choice_df_overall_points(bracket, march_madness_points_df):
    # At each comparison, choose the team that has the greater overall expected points from the tournament
    print('\t\tOverall points topology')

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)

    for team1, team2 in itertools.combinations(teams, 2):
        team1_total_chance = march_madness_points_df.at[team1, 'Total']
        team2_total_chance = march_madness_points_df.at[team2, 'Total']
        choice_df.at[team2, team1] = 1 if team1_total_chance > team2_total_chance else 0
        choice_df.at[team1, team2] = 1 if team2_total_chance > team1_total_chance else 0

    return choice_df


def create_choice_df_conditional_chance(bracket, march_madness_cond_chance_df):
    # At each comparison, choose the team that has the greater overall chance to win the tournament
    #   Assuming they have made it to the point in the bracket where the comparison takes place
    print('\t\tConditional chance topology')

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)

    rounds = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
    for team in teams:
        team_path = list(nx.all_simple_paths(bracket.to_graph(), team, 'Winner'))[0]

        for previous_bracket_round, bracket_location in zip(rounds, team_path[-6:]):
            sub_bracket = bracket.get_sub_bracket(bracket_location)
            team_chance = march_madness_cond_chance_df.at[team, previous_bracket_round]

            for possible_opponent in sub_bracket.get_possible_opponents(team):
                opp = possible_opponent.label
                opp_chance = march_madness_cond_chance_df.at[opp, previous_bracket_round]
                choice_df.at[opp, team] = 1 if team_chance > opp_chance else 0
                choice_df.at[team, opp] = 1 if opp_chance > team_chance else 0

    return choice_df


def create_choice_df_conditional_points(bracket, march_madness_cond_points_df):
    # At each comparison, choose the team that has the greater overall expected points from the tournament
    #   Assuming they have made it to the point in the bracket where the comparison takes place
    print('\t\tConditional points topology')

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)

    rounds = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
    for team in teams:
        team_path = list(nx.all_simple_paths(bracket.to_graph(), team, 'Winner'))[0]

        for previous_bracket_round, bracket_location in zip(rounds, team_path[-6:]):
            sub_bracket = bracket.get_sub_bracket(bracket_location)
            team_chance = march_madness_cond_points_df.at[team, previous_bracket_round]

            for possible_opponent in sub_bracket.get_possible_opponents(team):
                opp = possible_opponent.label
                opp_chance = march_madness_cond_points_df.at[opp, previous_bracket_round]
                choice_df.at[opp, team] = 1 if team_chance > opp_chance else 0
                choice_df.at[team, opp] = 1 if opp_chance > team_chance else 0

    return choice_df


def create_choice_df_overall_chance_sel(bracket, march_madness_chance_df):
    # At each comparison, choose the team that has the greater overall chance to win the tournament
    #   If a different team has already been chosen to reach a point in the bracket, assign their chance to 0
    print('\t\tSelective chance topology')

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)
    rounds = ['Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Winner']

    paths = [list(nx.all_simple_paths(bracket.to_graph(), team, 'Winner'))[0][-6:-1] for team in teams]
    path_df_cols = [bracket_round + ' Location' for bracket_round in rounds[:-1]]
    path_df = pd.DataFrame(paths, columns=path_df_cols, index=teams)
    combined_df = pd.concat([march_madness_chance_df, path_df], axis=1)

    for bracket_round in reversed(rounds[:-1]):
        round_winners = combined_df.groupby(bracket_round + ' Location')[bracket_round].transform('max') == \
                        combined_df[bracket_round]
        round_losers = [team for team, status in round_winners.items() if not status]
        round_winners = [team for team, status in round_winners.items() if status]

        for round_winner, round_loser in itertools.product(round_winners, round_losers):
            choice_df.at[round_loser, round_winner] = 1
            choice_df.at[round_winner, round_loser] = 0

    winner = march_madness_chance_df.idxmax()['Winner']
    for team in teams:
        if team == winner:
            continue
        choice_df.at[team, winner] = 1
        choice_df.at[winner, team] = 0

    pd.set_option('future.no_silent_downcasting', True)
    choice_df = choice_df.fillna(0)
    return choice_df


def create_choice_df_overall_points_sel(bracket, march_madness_points_df):
    # At each comparison, choose the team that has the greater expected points to gain moving forward
    print('\t\tSelective points topology')

    march_madness_points_df = march_madness_points_df.drop(columns='Total').cumsum(axis=1)
    march_madness_points_df['Round of 32 Gain'] = march_madness_points_df['Winner']
    march_madness_points_df['Sweet 16 Gain'] = march_madness_points_df['Winner'] - march_madness_points_df['Round of 32']
    march_madness_points_df['Elite 8 Gain'] = march_madness_points_df['Winner'] - march_madness_points_df['Sweet 16']
    march_madness_points_df['Final 4 Gain'] = march_madness_points_df['Winner'] - march_madness_points_df['Elite 8']
    march_madness_points_df['Championship Gain'] = march_madness_points_df['Winner'] - march_madness_points_df['Final 4']
    march_madness_points_df['Winner Gain'] = march_madness_points_df['Winner'] - march_madness_points_df['Championship']

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)
    rounds = ['Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Winner']

    march_madness_points_df = march_madness_points_df.drop(columns=rounds)
    march_madness_points_df = march_madness_points_df.rename(columns={r + ' Gain': r for r in rounds})

    paths = [list(nx.all_simple_paths(bracket.to_graph(), team, 'Winner'))[0][-6:-1] for team in teams]
    path_df_cols = [bracket_round + ' Location' for bracket_round in rounds[:-1]]
    path_df = pd.DataFrame(paths, columns=path_df_cols, index=teams)
    combined_df = pd.concat([march_madness_points_df, path_df], axis=1)

    for bracket_round in reversed(rounds[:-1]):
        round_winners = combined_df.groupby(bracket_round + ' Location')[bracket_round].transform('max') == \
                        combined_df[bracket_round]
        round_losers = [team for team, status in round_winners.items() if not status]
        round_winners = [team for team, status in round_winners.items() if status]

        for round_winner, round_loser in itertools.product(round_winners, round_losers):
            choice_df.at[round_loser, round_winner] = 1
            choice_df.at[round_winner, round_loser] = 0

    winner = march_madness_points_df.idxmax()['Winner']
    for team in teams:
        if team == winner:
            continue
        choice_df.at[team, winner] = 1
        choice_df.at[winner, team] = 0

    pd.set_option('future.no_silent_downcasting', True)
    choice_df = choice_df.fillna(0)
    return choice_df


def create_choice_df_points_loss_min(bracket, march_madness_points_df):
    # For each round, calculate the total expected points missed by selecting each team
    # Then select the team that has the minimum amount of expected points lost
    print('\t\tLoss minimizing topology')

    march_madness_points_df = march_madness_points_df.copy()

    teams = tournament.get_bracket_teams(bracket)
    choice_df = pd.DataFrame(index=teams, columns=teams)
    rounds = ['Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Winner']
    loss_df = pd.DataFrame(index=teams, columns=rounds)

    for bracket_round in reversed(rounds):
        selection_loss = {team: tournament.get_selection_loss(bracket, march_madness_points_df, team, bracket_round)
                          for team in teams}
        for team, loss in selection_loss.items():
            loss_df.at[team, bracket_round] = loss

        paths = [list(nx.all_simple_paths(bracket.to_graph(), team, 'Winner'))[0][-6:] for team in teams]
        path_df_cols = [bracket_round + ' Location' for bracket_round in rounds]
        path_df = pd.DataFrame(paths, columns=path_df_cols, index=teams)
        combined_df = pd.concat([loss_df, path_df], axis=1)

        round_winners = combined_df.groupby(bracket_round + ' Location')[bracket_round].transform('min') == \
                        combined_df[bracket_round]
        round_losers = [team for team, status in round_winners.items() if not status]
        round_winners = [team for team, status in round_winners.items() if status]

        for round_winner, round_loser in itertools.product(round_winners, round_losers):
            choice_df.at[round_loser, round_winner] = 1
            choice_df.at[round_winner, round_loser] = 0

        for round_loser in round_losers:
            march_madness_points_df.at[round_loser, bracket_round] = 0

    pd.set_option('future.no_silent_downcasting', True)
    choice_df = choice_df.fillna(0)
    return choice_df


def create_topology(team_df, choice_df):
    pd.set_option('future.no_silent_downcasting', True)
    choice_df = choice_df.fillna(0)
    teams = team_df.index

    num_better_than = {team: sum(choice_df[team]) for team in teams}
    num_better_than = {team: count for team, count in sorted(num_better_than.items(), key=lambda t: t[1], reverse=True)}
    ordered_teams = list(num_better_than.keys())

    team_df = team_df.reindex(ordered_teams)
    return team_df


def topology_bracket(bracket, topology):
    topology = topology.copy()
    team_ordering = list(topology.index)

    paths = [list(nx.all_simple_paths(bracket.to_graph(), team, 'Winner'))[0][:-1] for team in team_ordering]
    paths = [path[-5:] for path in paths]
    path_df = pd.DataFrame(paths,
                           index=team_ordering,
                           columns=['Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship'])

    round_32_winners = list(path_df.drop_duplicates(subset=['Round of 32'], keep='first').index)
    sweet_16_winners = list(path_df.drop_duplicates(subset=['Sweet 16'], keep='first').index)
    elite_8_winners = list(path_df.drop_duplicates(subset=['Elite 8'], keep='first').index)
    final_4_winners = list(path_df.drop_duplicates(subset=['Final 4'], keep='first').index)
    semi_winners = list(path_df.drop_duplicates(subset=['Championship'], keep='first').index)
    winner = [team_ordering[0]]

    topology_wins = {team: 0 for team in team_ordering}
    topology_wins.update({team: 1 for team in round_32_winners})
    topology_wins.update({team: 2 for team in sweet_16_winners})
    topology_wins.update({team: 3 for team in elite_8_winners})
    topology_wins.update({team: 4 for team in final_4_winners})
    topology_wins.update({team: 5 for team in semi_winners})
    topology_wins.update({team: 6 for team in winner})

    topology['Team'] = topology.index
    topology['Round of 32'] = topology.apply(lambda r: 0 if r['Team'] not in round_32_winners else r['Round of 32'],
                                             axis=1)
    topology['Sweet 16'] = topology.apply(lambda r: 0 if r['Team'] not in sweet_16_winners else r['Sweet 16'], axis=1)
    topology['Elite 8'] = topology.apply(lambda r: 0 if r['Team'] not in elite_8_winners else r['Elite 8'], axis=1)
    topology['Final 4'] = topology.apply(lambda r: 0 if r['Team'] not in final_4_winners else r['Final 4'], axis=1)
    topology['Championship'] = topology.apply(lambda r: 0 if r['Team'] not in semi_winners else r['Championship'],
                                              axis=1)
    topology['Winner'] = topology.apply(lambda r: 0 if r['Team'] not in winner else r['Winner'], axis=1)

    topology['Total'] = topology.apply(lambda r: r['Round of 32'] +
                                                 r['Sweet 16'] +
                                                 r['Elite 8'] +
                                                 r['Final 4'] +
                                                 r['Championship'] +
                                                 r['Winner'], axis=1)

    topology['Wins'] = topology.apply(lambda r: topology_wins.get(r['Team'], 0), axis=1)
    topology = topology.sort_values(by=['Winner',
                                        'Championship',
                                        'Final 4',
                                        'Elite 8',
                                        'Sweet 16',
                                        'Round of 32',
                                        'Team',
                                        'Wins'], kind='mergesort', ascending=False)

    drop_columns = ['Team', 'Wins']
    topology = topology.drop(columns=drop_columns)

    topology.loc['Total'] = topology.sum()
    return topology

import math
import copy
import networkx as nx
from scipy.stats import binom
import json
from itertools import islice


class Bracket:
    def __init__(self, label, left=None, right=None, series_length=1):
        self.label = label
        self.left = left
        self.right = right
        self.is_leaf = left is None and right is None
        self.series_length = series_length

    def copy(self):
        return copy.deepcopy(self)

    def size(self):
        return 1 if self.is_leaf else self.left.size() + self.right.size()

    def get_leaves(self):
        return [self] if self.is_leaf else self.left.get_leaves() + self.right.get_leaves()

    def to_graph(self):
        graph = nx.DiGraph()
        if self.is_leaf:
            graph.add_node(self.label)
        else:
            left_graph = self.left.to_graph()
            right_graph = self.right.to_graph()

            graph.add_edge(self.left.label, self.label)
            graph.add_edge(self.right.label, self.label)

            graph = nx.compose(graph, left_graph)
            graph = nx.compose(graph, right_graph)
        return graph

    def get_possible_opponents(self, label):
        if self.is_leaf:
            return []
        if label not in [leaf.label for leaf in self.get_leaves()]:
            return []

        left_side = self.left.get_leaves()
        left_side = [] if label in [b.label for b in left_side] else left_side
        right_side = self.right.get_leaves()
        right_side = [] if label in [b.label for b in right_side] else right_side

        return left_side + right_side

    def get_all_opponents(self, label):
        team_path = list(nx.all_simple_paths(self.to_graph(), label, 'Winner'))[0]
        team_path = team_path[1:]

        def rename_round(r_name):
            if r_name == 'Winner':
                return r_name
            tokens = r_name.split(' ')
            return ' '.join(tokens[:-1])

        round_names = [rename_round(bracket_round) for bracket_round in team_path]

        all_opponents = {'Play In': []}
        for round_name, bracket_round in zip(round_names, team_path):
            sub_bracket = self.get_sub_bracket(bracket_round)
            possible_opponents = [leaf.label for leaf in sub_bracket.get_possible_opponents(label)]
            all_opponents[round_name] = possible_opponents

        all_opponents['Round of 32'] = all_opponents.get('Round of 32', []) + all_opponents.get('Play In', [])
        del all_opponents['Play In']

        return all_opponents

    def get_victory_chance(self, label, chance_df):
        # Get all possible teams that can reach this point in the bracket
        leaf_labels = {b.label for b in self.get_leaves()}

        # If the team in question is not one of them, they have a 0% chance
        if label not in leaf_labels:
            return 0

        # If the team in question is the only one, they have a 100% chance
        if self.is_leaf:
            return 1

        # The teams chance to reach this point of the bracket is their chance to win the previous point
        reach_chance = self.left.get_victory_chance(label, chance_df) + self.right.get_victory_chance(label, chance_df)

        team_leaf = [b for b in self.get_leaves() if b.label == label][0]
        possible_victory_chances = []
        for other_team in self.get_possible_opponents(label):
            other_label = other_team.label
            team_chance, opp_chance = self.get_best_of(team_leaf, other_team, chance_df)
            opp_reach_chance = (self.left.get_victory_chance(other_label, chance_df) +
                                self.right.get_victory_chance(other_label, chance_df))

            possible_victory_chances.append(team_chance * opp_reach_chance)

        victory_chance = reach_chance * sum(possible_victory_chances)

        return victory_chance

    def get_best_of(self, team1, team2, chances_df):
        p = chances_df.at[team2.label, team1.label]
        required_wins = int(math.ceil(self.series_length / 2.0))

        team1_chance = binom.cdf(self.series_length - required_wins, self.series_length, 1 - p)
        team2_chance = binom.cdf(self.series_length - required_wins, self.series_length, p)

        return team1_chance, team2_chance

    def get_sub_bracket(self, label):
        if self.label == label:
            return self
        if not self.is_leaf:
            left = self.left.get_sub_bracket(label)
            right = self.right.get_sub_bracket(label)
            if left:
                return left
            if right:
                return right
        return None


def mm_bracket(config):
    with open(config.get('resource_locations').get('name_to_school'), 'r') as f:
        name_map = json.load(f)

    with open(config.get('resource_locations').get('bracket'), 'r') as f:
        bracket_structure = json.load(f)

    num_play_in = 0
    games = bracket_structure.get('Games')

    r32_matchups = list()
    for game_num, game in enumerate(games):
        team1 = game.get('team1')
        team2 = game.get('team2')

        if team1 not in name_map.values():
            print('Incorrect team in bracket creation:', team1)
            return

        if isinstance(team2, str):
            if team2 not in name_map.values():
                print('Incorrect team in bracket creation:', team2)
                return

            matchup = Bracket('Round of 32 ' + str(game_num + 1), left=Bracket(team1), right=Bracket(team2))
        else:
            num_play_in = num_play_in + 1
            play_in_team1 = team2.get('team1')
            play_in_team2 = team2.get('team2')

            if play_in_team1 not in name_map.values():
                print('Incorrect team in bracket creation:', play_in_team1)
                return

            if play_in_team2 not in name_map.values():
                print('Incorrect team in bracket creation:', play_in_team2)
                return

            play_in_matchup = Bracket('Play In ' + str(num_play_in),
                                      left=Bracket(play_in_team1), right=Bracket(play_in_team2))
            matchup = Bracket('Round of 32 ' + str(game_num + 1), left=Bracket(team1), right=play_in_matchup)
        r32_matchups.append(matchup)

    sweet_16_count = 1
    s16_matchups = list()
    for matchup1, matchup2 in batched(r32_matchups, n=2, strict=True):
        matchup = Bracket('Sweet 16 ' + str(sweet_16_count), left=matchup1, right=matchup2)
        sweet_16_count = sweet_16_count + 1
        s16_matchups.append(matchup)

    elite_8_count = 1
    e8_matchups = list()
    for matchup1, matchup2 in batched(s16_matchups, n=2, strict=True):
        matchup = Bracket('Elite 8 ' + str(elite_8_count), left=matchup1, right=matchup2)
        elite_8_count = elite_8_count + 1
        e8_matchups.append(matchup)

    final_4_count = 1
    f4_matchups = list()
    for matchup1, matchup2 in batched(e8_matchups, n=2, strict=True):
        matchup = Bracket('Final 4 ' + str(final_4_count), left=matchup1, right=matchup2)
        final_4_count = final_4_count + 1
        f4_matchups.append(matchup)

    championship_count = 1
    c2_matchups = list()
    for matchup1, matchup2 in batched(f4_matchups, n=2, strict=True):
        matchup = Bracket('Championship ' + str(championship_count), left=matchup1, right=matchup2)
        championship_count = championship_count + 1
        c2_matchups.append(matchup)

    if len(c2_matchups) != 2:
        print('Incorrect number of teams in bracket')
        return

    champ = Bracket('Winner', left=c2_matchups[0], right=c2_matchups[1])
    return champ


def batched(iterable, n, *, strict=False):
    # from itertools 3.13
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

import numpy as np


def get_tiers(bt_df, ranking_column='BT'):
    tier_cutoffs = list(reversed(np.histogram_bin_edges(list(bt_df[ranking_column]), bins='fd')))[1:]

    bt_df['Tier'] = bt_df.apply(lambda r:
                                tier_cutoffs.index(max([tier for tier in tier_cutoffs if r[ranking_column] >= tier]))
                                + 1, axis=1)

    return bt_df


def get_chance_to_beat_team(chances_df, team, opponent):
    return chances_df.at[opponent, team]

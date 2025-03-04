import json
import math
import warnings

import PIL
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use('TkAgg')


def show_off_def(config, model_df, location):
    warnings.filterwarnings("ignore")

    sns.set(style="ticks")

    # Format and title the graph
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title('')
    ax.set_xlabel('Adjusted Points For')
    ax.set_ylabel('Adjusted Points Against')
    ax.set_facecolor('#FAFAFA')

    with open(config.get('resource_locations').get('school_to_logo'), 'r') as f:
        file_map = json.load(f)

    images = {team: PIL.Image.open('resources/logos/' + file_map.get(team, team) + '.png')
              for team, row in model_df.iterrows()}

    margin = 1
    min_x = model_df['Adj. Offense'].min() - margin
    max_x = model_df['Adj. Offense'].max() + margin

    min_y = model_df['Adj. Defense'].min() - margin
    max_y = model_df['Adj. Defense'].max() + margin

    ax = plt.gca()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.set_aspect(aspect=0.3, adjustable='datalim')

    for team in model_df.index:
        xa = model_df.at[team, 'Adj. Offense']
        ya = model_df.at[team, 'Adj. Defense']

        offset = .6
        img_alpha = .8
        ax.imshow(images.get(team), extent=(xa - offset, xa + offset, ya + offset, ya - offset), alpha=img_alpha)

    vert_mean = model_df['Adj. Offense'].mean()
    horiz_mean = model_df['Adj. Defense'].mean()
    plt.axvline(x=vert_mean, color='r', linestyle='--', alpha=.5)
    plt.axhline(y=horiz_mean, color='r', linestyle='--', alpha=.5)

    offset_dist = 5 * math.sqrt(2)
    offsets = set(np.arange(0, 75, offset_dist))
    offsets = offsets.union({-offset for offset in offsets})

    for offset in [horiz_mean + offset for offset in offsets]:
        plt.axline(xy1=(vert_mean, offset), slope=1, alpha=.1)

    # Show the graph
    plt.savefig(config.get('output_locations').get(location), dpi=300)


def plot_sankey(config, chance_df, bracket):
    teams = bracket.get_sub_bracket('Final 4 2').get_leaves()
    teams = [l.label for l in teams]

    chance_df = chance_df.loc[teams]
    chance_df = chance_df[['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4']]

    i = 0

from prettytable import PrettyTable

from app import team_info
from app import team_ranking
from app import tournament


def print_team_rankings(config, bracket, topo_df, games_graph, model_num, ranking_column='BT'):
    teams = tournament.get_bracket_teams(bracket)
    topo_df = team_ranking.get_tiers(topo_df, ranking_column=ranking_column)
    topo_df = topo_df.loc[teams]

    topo_df = topo_df.sort_values(by=ranking_column, kind='mergesort', ascending=False)
    topo_df = topo_df.reset_index(names='Team')
    topo_df = topo_df.reset_index(names='Rank')
    topo_df['Rank'] = topo_df.apply(lambda r: r['Rank'] + 1, axis=1)
    columns = list(topo_df.columns)
    if 'Var' in columns:
        topo_df = topo_df.drop(columns=['Var'])
    if 'Model' in columns:
        topo_df = topo_df.drop(columns=['Model'])
    if 'Intercept' in columns:
        topo_df = topo_df.drop(columns=['Intercept'])
    if 'Dispersion' in columns:
        topo_df = topo_df.drop(columns=['Dispersion'])
    if 'Neutral Coef' in columns:
        topo_df = topo_df.drop(columns=['Neutral Coef'])
    if 'Home Coef' in columns:
        topo_df = topo_df.drop(columns=['Home Coef'])

    topo_df['Record'] = topo_df.apply(lambda r: str(games_graph.in_degree(r['Team'])) + ' - ' +
                                                str(games_graph.out_degree(r['Team'])), axis=1)

    columns = list(topo_df.columns)
    topo_df = topo_df[columns[:-2] + ['Record', 'Tier']]

    columns = list(topo_df.columns)
    table = PrettyTable(columns)
    table.float_format = '0.3'

    for index, row in topo_df.iterrows():
        table_row = list()
        table_row.extend([item for i, item in list(row.items())])
        table.add_row(table_row)

    ranking_log_map = {1: 'Overall Team Rankings: Game by Game Matchup',
                       2: 'Team Rankings Adjusting for Home Court Advantage',
                       3: 'Team Rankings Adjusting for Home Court Advantage and Road Disadvantage',
                       4: 'Team Rankings Adjusting for Conference',
                       5: 'Team Rankings Based on Neg Binomial Adjusted Score',
                       6: 'Team Rankings Based on Generalized Poisson Adjusted Score'}

    with open(config.get('output_locations').get('team_rankings').format(str(model_num)), 'w') as f:
        f.write(ranking_log_map.get(model_num, 'Overall Team Rankings: Game by Game Matchup') + '\n')
        table_str = str(table)
        f.write(table_str)
        f.close()


def print_team_chances(config, topo_df, model_num):
    school_mapping = team_info.map_full_names_to_school(config)
    topo_df.loc['TOTAL'] = topo_df.sum() / 100.0
    topo_df = topo_df.reset_index(names='Team')

    topo_df['Team'] = topo_df.apply(lambda r: school_mapping.get(r['Team'], r['Team']), axis=1)

    columns = list(topo_df.columns)
    table = PrettyTable(columns)
    table.float_format = '0.3'

    for index, row in topo_df.iterrows():
        table_row = list()
        table_row.append(row['Team'])
        if row['Team'] != 'TOTAL':
            table_row.extend([f'{row[column] * 100:.1f}' + '%' for column in columns[1:]])
        else:
            table_row.extend([round(row[column] * 100) for column in columns[1:]])
        table.add_row(table_row)

    with open(config.get('output_locations').get('round_chances').format(str(model_num)), 'w') as f:
        f.write('Chance For Each Team To Make Each Round\n')
        table_str = str(table)
        f.write(table_str)
        f.close()


def print_team_points(config, topo_df, model_num, table_str, table_path):
    school_mapping = team_info.map_full_names_to_school(config)
    topo_df = topo_df.reset_index(names='Team')

    topo_df['Team'] = topo_df.apply(lambda r: school_mapping.get(r['Team'], r['Team']), axis=1)

    columns = list(topo_df.columns)
    table = PrettyTable(columns)
    table.float_format = '0.3'

    for index, row in topo_df.iterrows():
        table_row = list()
        table_row.extend([item for i, item in list(row.items())])
        table.add_row(table_row)

    with open('output/Model ' + str(model_num) + '/' + table_path, 'w') as f:
        f.write(table_str)
        table_str = str(table)
        f.write(table_str)
        f.close()


def print_conference_rankings(config, bracket, conf_bt_df):
    teams = tournament.get_bracket_teams(bracket)
    conf_bt_df = conf_bt_df.loc[teams]

    table = PrettyTable(['Rank', 'Conference', 'BT'])
    table.float_format = '0.3'

    bt_df = conf_bt_df[['Conference', 'Conference BT']]
    bt_df = bt_df.drop_duplicates(subset=['Conference', 'Conference BT'])
    bt_df = bt_df.sort_values(by='Conference BT', kind='mergesort', ascending=False)
    bt_df = bt_df.reset_index()

    for index, row in bt_df.iterrows():
        table_row = list()

        table_row.append(index + 1)
        table_row.append(row['Conference'])
        table_row.append(row['Conference BT'])

        table.add_row(table_row)

    with open(config.get('output_locations').get('model4_conf_rankings'), 'w') as f:
        f.write('Conference Rankings\n')
        table_str = str(table)
        f.write(table_str)
        f.close()


def print_intra_conf_rankings(config, conf_bt_df, games_graph):
    conf_bt_df = conf_bt_df.sort_values(by=['Conference BT', 'Team BT'], kind='mergesort', ascending=[False, False])
    conf_ranking_df = conf_bt_df[['Conference', 'Conference BT']]
    conf_ranking_df = conf_ranking_df.drop_duplicates(subset=['Conference', 'Conference BT'])
    conf_ranking_df = conf_ranking_df.sort_values(by='Conference BT', kind='mergesort', ascending=False)
    conf_ranking_df = conf_ranking_df.reset_index()

    with open(config.get('output_locations').get('model4_intra_conf_rankings'), 'w') as f:
        f.write('Team Rankings Within Conference')
        f.close()

    for index, conference in conf_ranking_df['Conference'].items():
        conf_df = conf_bt_df.loc[conf_bt_df['Conference'] == conference]

        table = PrettyTable(['Rank', 'Team', 'Record', 'BT'])
        table.float_format = '0.3'

        school_mapping = team_info.map_full_names_to_school(config)

        for index, row_info in enumerate(conf_df.iterrows()):
            table_row = list()
            team, row = row_info

            wins = games_graph.in_degree(team)
            losses = games_graph.out_degree(team)

            if conference == 'Independent' and wins + losses < 20:
                continue

            table_row.append(index + 1)
            table_row.append(school_mapping.get(team, team))

            table_row.append(str(wins) + ' - ' + str(losses))
            table_row.append(row['Team BT'])

            table.add_row(table_row)

        with open(config.get('output_locations').get('model4_intra_conf_rankings'), 'a') as f:
            f.write('\n\n' + conference + ' Rankings\n')
            table_str = str(table)
            f.write(table_str)
            f.close()

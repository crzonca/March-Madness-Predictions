import json

from app import bracket
from app import decision_functions
from app import table_printer
from app import team_season
from app import tournament
from app import validation
from app.models import model1 as m1
from app.models import model2 as m2
from app.models import model3 as m3
from app.models import model4 as m4
from app.models import model5 as m5
from app.models import model6 as m6
from app.models import model7 as m7


def march_madness(model=1):
    with open('resources/config.json', 'r') as f:
        config = json.load(f)

    with open(config.get('resource_locations').get('name_to_school'), 'r') as f:
        name_map = json.load(f)

    br = bracket.mm_bracket(config)

    if not 1 <= model <= 7:
        model = 1

    model_desc_map = {1: 'Model 1: Pure BT',
                      2: 'Model 2: BT with Home Court Advantage',
                      3: 'Model 3: BT with Home Court Advantage and Road Disadvantage',
                      4: 'Model 4: BT with Offset for Conference Strength',
                      5: 'Model 5: Negative Binomial Regression',
                      6: 'Model 6: Generalized Poisson Regression',
                      7: 'Model 7: Generalized Poisson Regression with Home Court Advantage'}
    print(model_desc_map.get(model))

    graph = team_season.get_games_graph(config)
    home_winner_graph, neut_winner_graph, away_winner_graph = team_season.get_location_games_graph(config)

    # data_valid = validation.validate_data(config, graph)
    # if not data_valid:
    #     return

    ranking_func_map = {1: m1.get_ranking_and_chance,
                        2: m2.get_ranking_and_chance,
                        3: m3.get_ranking_and_chance,
                        4: m4.get_ranking_and_chance,
                        5: m5.get_ranking_and_chance,
                        6: m6.get_ranking_and_chance,
                        7: m7.get_ranking_and_chance}

    ranking_func_args_map = {1: (br, graph,),
                             2: (br, graph, home_winner_graph, neut_winner_graph, away_winner_graph),
                             3: (br, graph, home_winner_graph, neut_winner_graph, away_winner_graph),
                             4: (br, graph, config,),
                             5: (br, config,),
                             6: (br, config,),
                             7: (br, config,)}

    # Get each teams ranking according to the model
    ranking_function = ranking_func_map.get(model, m1.get_ranking_and_chance)
    ranking_func_args = ranking_func_args_map.get(model, (graph,))

    print('\tRanking teams and getting individual match up chances')
    ranking_df, chances_df = ranking_function(*ranking_func_args)

    if model == 4:
        table_printer.print_conference_rankings(config, br, ranking_df)
        table_printer.print_intra_conf_rankings(config, ranking_df, graph)

    ranking_column_map = {1: 'BT',
                          2: 'BT',
                          3: 'BT',
                          4: 'BT',
                          5: 'Completeness',
                          6: 'Completeness',
                          7: 'Completeness'}

    # 1
    table_printer.print_team_rankings(config, br, ranking_df, graph,
                                      model_num=model, ranking_column=ranking_column_map.get(model))

    # Get each teams chance to win against every other team and their expected tournament points according to the model
    if model == 2 or model == 3:
        home_chances_df, chances_df, away_chances_df = chances_df
    chances_df = chances_df.rename(index=name_map, columns=name_map)

    print('\tGetting tournament chances and expected points')
    march_madness_chance_df, march_madness_points_df = tournament.calculate_overall_chances_points(br, chances_df)
    march_madness_chance_df, march_madness_points_df = tournament.update_tournament_chances(config,
                                                                                            march_madness_chance_df)

    print('\tGetting conditional tournament chances and expected points')
    mm_cond_chance_df, mm_cond_points_df = tournament.calculate_conditional_chances_points(march_madness_chance_df)

    # 2
    table_printer.print_team_chances(config, march_madness_chance_df.copy(), model)

    # 3
    table_printer.print_team_points(config, march_madness_points_df.copy(), model,
                                    'Expected Points For Each Team From Making Each Round\n',
                                    '3 Expected Points.txt')

    # Get Topologies
    print('\tBuilding topologies')

    # 4
    points_topology_loss = decision_functions.complete_bracket(br, march_madness_points_df, march_madness_points_df,
                                                               decision_functions.create_choice_df_points_loss_min)
    table_printer.print_team_points(config, points_topology_loss.copy(), model,
                                    'Expected Points For Each Team Taking the Team That Minimizes The Expected Points '
                                    'Lost For Each Round\n',
                                    '4 Loss Minimizing Points Topology.txt')

    # 5
    better_team_topology = decision_functions.complete_bracket(br, ranking_df, march_madness_points_df,
                                                               decision_functions.create_choice_df_better_team)
    table_printer.print_team_points(config, better_team_topology.copy(), model,
                                    'Expected Points For Each Team Taking the Better Team Each Round\n',
                                    '5 Better Team Topology.txt')

    # 6
    chance_topology = decision_functions.complete_bracket(br, march_madness_chance_df, march_madness_points_df,
                                                          decision_functions.create_choice_df_overall_chance)
    table_printer.print_team_points(config, chance_topology.copy(), model,
                                    'Expected Points For Each Team Taking the '
                                    'Team With the Highest Overall Victory Chance Each Round\n',
                                    '6 Overall Chance Topology.txt')

    # 7
    points_topology = decision_functions.complete_bracket(br, march_madness_points_df, march_madness_points_df,
                                                          decision_functions.create_choice_df_overall_points)
    table_printer.print_team_points(config, points_topology.copy(), model,
                                    'Expected Points For Each Team Taking the '
                                    'Team With the Highest Overall Expected Points Each Round\n',
                                    '7 Overall Points Topology.txt')

    # 8
    chance_topology_cond = decision_functions.complete_bracket(br, mm_cond_chance_df, march_madness_points_df,
                                                               decision_functions.create_choice_df_conditional_chance)
    table_printer.print_team_points(config, chance_topology_cond.copy(), model,
                                    'Expected Points For Each Team Taking the Team With the Highest Victory Chance '
                                    'Each Round, Assuming They Have Made it that far\n',
                                    '8 Conditional Chance Topology.txt')

    # 9
    points_topology_cond = decision_functions.complete_bracket(br, mm_cond_points_df, march_madness_points_df,
                                                               decision_functions.create_choice_df_conditional_points)
    table_printer.print_team_points(config, points_topology_cond.copy(), model,
                                    'Expected Points For Each Team Taking the Team With the Highest Expected Points '
                                    'Each Round, Assuming They Have Made it that far\n',
                                    '9 Conditional Points Topology.txt')

    # 10
    chance_topology_sel = decision_functions.complete_bracket(br, march_madness_chance_df, march_madness_points_df,
                                                              decision_functions.create_choice_df_overall_chance_sel)
    table_printer.print_team_points(config, chance_topology_sel.copy(), model,
                                    'Expected Points For Each Team Taking the Team With the Highest Overall Chance '
                                    'Each Round, Eliminating All Others That Cannot Continue\n',
                                    '10 Selective Chance Topology.txt')

    # 11
    points_topology_sel = decision_functions.complete_bracket(br, march_madness_points_df, march_madness_points_df,
                                                              decision_functions.create_choice_df_overall_points_sel)
    table_printer.print_team_points(config, points_topology_sel.copy(), model,
                                    'Expected Points For Each Team Taking the Team With the Highest Expected Points '
                                    'Each Round, Eliminating All Others That Cannot Continue\n',
                                    '11 Selective Points Topology.txt')

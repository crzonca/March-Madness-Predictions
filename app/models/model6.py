import itertools
import json
import warnings

import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata
from statsmodels.discrete.discrete_model import GeneralizedPoisson
from statsmodels.iolib.smpickle import load_pickle

from app import tournament
from app import visualization


# Model 6: Generalized Poisson Regression
def get_ranking_and_chance(bracket, config, verbose=False, use_all=False):
    ranking_df, gen_poisson_model, possessions_model = get_team_ranking(bracket, config, verbose=verbose)
    chance_df = create_chance_df(bracket,
                                 config,
                                 gen_poisson_model,
                                 possessions_model,
                                 use_all=use_all)
    return ranking_df, chance_df


def fit_gen_poisson(df, config, verbose=False, use_persisted=True):
    if use_persisted:
        return load_pickle(config.get('output_locations').get('model6_pickle'))
    points_df = df[['Team', 'Opponent', 'Points', 'Possessions']]
    points_df = points_df.rename(columns={'Team': 'Offense',
                                          'Opponent': 'Defense'})

    response = points_df['Points']
    explanatory = pd.get_dummies(points_df[['Offense', 'Defense']], dtype=int)
    explanatory = sm.add_constant(explanatory)

    gen_poisson = GeneralizedPoisson(endog=response,
                                     exog=explanatory,
                                     exposure=points_df['Possessions'],
                                     p=1)

    gen_poisson_model = gen_poisson.fit_regularized(method='l1',
                                                    maxiter=int(1e7),
                                                    alpha=0.1,
                                                    disp=0)

    with open(config.get('output_locations').get('model6_summary'), 'w') as f:
        f.write(str(gen_poisson_model.summary()))
        f.close()
    gen_poisson_model.save(config.get('output_locations').get('model6_pickle'))

    if verbose:
        print(gen_poisson_model.summary())

    return gen_poisson_model


def get_gen_poisson_model(config, verbose=False):
    df = pd.read_csv(config.get('resource_locations').get('stat_lines'))

    with open(config.get('resource_locations').get('name_to_school'), 'r') as f:
        name_map = json.load(f)

    df['Team'] = df.apply(lambda r: name_map.get(r['Team'], r['Team']), axis=1)
    df['Opponent'] = df.apply(lambda r: name_map.get(r['Opponent'], r['Opponent']), axis=1)

    df['GameDay'] = pd.to_datetime(df['GameDay'])
    df['2FM'] = df.apply(lambda r: r['FGM'] - r['3FM'], axis=1)
    df['2FA'] = df.apply(lambda r: r['FGA'] - r['3FA'], axis=1)

    df['Points'] = df.apply(lambda r: r['3FM'] * 3 + r['2FM'] * 2 + r['FTM'], axis=1)
    df['Possessions'] = df.apply(lambda r: r['FGA'] - r['OREB'] + r['TO'] + int(r['FTA'] / 2), axis=1)

    df = df[['GameDay', 'Team', 'Opponent', 'Points', 'FGM', 'FGA', '2FM', '2FA',
             '3FM', '3FA', 'FTM', 'FTA', 'OREB', 'TO', 'Possessions']]
    df = df.loc[(df['Team'] != 'ND1') & (df['Opponent'] != 'ND1')]
    df = df.loc[(~pd.isna(df['Team'])) & (~pd.isna(df['Opponent']))]
    df = df.dropna()

    model = fit_gen_poisson(df, config, verbose=verbose, use_persisted=config.get('use_persisted_poisson'))
    return model, df


def get_team_ranking(bracket, config, verbose=True):
    model, df = get_gen_poisson_model(config, verbose=verbose)
    possessions_model = fit_possessions(df, config, verbose=verbose)

    team_names = set([param.replace('Offense_', '').replace('Defense_', '')
                      for param in model.params.index]) - {'const', 'alpha'}

    off_coefs = {team: model.params['Offense_' + team] for team in team_names}
    def_coefs = {team: model.params['Defense_' + team] for team in team_names}
    intercept = model.params['const']
    alpha = model.params['alpha']

    rows = list()
    mean_possessions = df['Possessions'].mean()
    explanatory = pd.get_dummies(df[['Team', 'Opponent']], dtype=int)
    explanatory = sm.add_constant(explanatory)
    for team in off_coefs.keys():
        off_coef = off_coefs.get(team, 0)
        def_coef = def_coefs.get(team, 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            off_prediction_series = pd.Series(index=explanatory.columns)
            off_prediction_series.at['const'] = 1.0
            off_prediction_series.at['Team_' + team] = 1.0
            off_prediction_series = off_prediction_series.fillna(0.0)

            def_prediction_series = pd.Series(index=explanatory.columns)
            def_prediction_series.at['const'] = 1.0
            def_prediction_series.at['Opponent_' + team] = 1.0
            def_prediction_series = def_prediction_series.fillna(0.0)

        off_dist = model.get_distribution(off_prediction_series, exposure=mean_possessions)
        off_adj = float(off_dist.mean())

        def_dist = model.get_distribution(def_prediction_series, exposure=mean_possessions)
        def_adj = float(def_dist.mean())

        adj_diff = off_adj - def_adj

        rows.append({'Team': team,
                     'Intercept': intercept,
                     'Alpha': alpha,
                     'Off. Coef': off_coef,
                     'Def. Coef': def_coef,
                     'Completeness': off_coef - def_coef,
                     'Adj. Offense': off_adj,
                     'Adj. Defense': def_adj,
                     'Adj. Diff': adj_diff})

    results_df = pd.DataFrame(rows)
    results_df = results_df.set_index('Team', drop=True)
    results_df = results_df.sort_values(by='Completeness', ascending=False)
    results_df['Rank'] = rankdata(-results_df['Completeness'])
    results_df['Offense Rank'] = rankdata(-results_df['Off. Coef'])
    results_df['Defense Rank'] = rankdata(results_df['Def. Coef'])

    results_df.to_csv(config.get('output_locations').get('model6_rankings'))
    results_df = results_df.drop(columns=['Rank', 'Offense Rank', 'Defense Rank'])

    visualization.show_off_def(config,
                               results_df.loc[[leaf.label for leaf in bracket.get_leaves()]],
                               'model6_offense_defense')

    return results_df, model, possessions_model


def fit_possessions(df, config, verbose=False):
    y = df['Possessions']
    x = pd.get_dummies(df[['Team', 'Opponent']], dtype=int)
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()

    if verbose:
        if results.f_pvalue > .05:
            print('OLS model is not a good fit')
        print(results.summary())

    with open(config.get('output_locations').get('model6_possessions_summary'), 'w') as f:
        f.write(str(results.summary()))
        f.close()
    results.save(config.get('output_locations').get('model6_possessions_pickle'))

    if verbose:
        print(results.summary())

    return results


def predict_possessions(possessions_model, team1, team2):
    intercept = possessions_model.params['const']
    team1_off_coef = possessions_model.params['Team_' + team1]
    team1_def_coef = possessions_model.params['Opponent_' + team1]
    team2_off_coef = possessions_model.params['Team_' + team2]
    team2_def_coef = possessions_model.params['Opponent_' + team2]

    team1_drives = intercept + team1_off_coef + team2_def_coef
    team2_drives = intercept + team2_off_coef + team1_def_coef
    average_drives = (team1_drives + team2_drives) / 2
    return average_drives


def get_victory_chance(dist1, dist2):
    max_points = 200

    team2_chances = dist2.pmf([score for score in range(max_points)])
    team1_chances = dist1.pmf([score for score in range(max_points)])
    team1_better_chances = dist1.sf([score for score in range(max_points)])

    win_chances = team2_chances * team1_better_chances
    tie_chances = team2_chances * team1_chances

    win_chance = sum(win_chances)
    tie_chance = sum(tie_chances)
    win_chance = win_chance / (1 - tie_chance)

    return win_chance


def create_chance_df(bracket, config, model, possessions_model, use_all=False):
    team_names = set([param.replace('Offense_', '').replace('Defense_', '')
                      for param in model.params.index]) - {'const'} - {'alpha'}
    teams = team_names if use_all else tournament.get_bracket_teams(bracket)
    chances_df = pd.DataFrame(index=teams, columns=teams)

    prediction_index = [param for param in model.params.index]
    for team1, team2 in itertools.combinations(teams, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            possessions = predict_possessions(possessions_model, team1, team2)

            t1_prediction_series = pd.Series(index=prediction_index)
            t1_prediction_series.at['const'] = 1.0
            t1_prediction_series.at['Offense_' + team1] = 1.0
            t1_prediction_series.at['Defense_' + team2] = 1.0
            t1_prediction_series = t1_prediction_series.fillna(0.0)

            t2_prediction_series = pd.Series(index=prediction_index)
            t2_prediction_series.at['const'] = 1.0
            t2_prediction_series.at['Offense_' + team2] = 1.0
            t2_prediction_series.at['Defense_' + team1] = 1.0
            t2_prediction_series = t2_prediction_series.fillna(0.0)

        dist1 = model.get_distribution(t1_prediction_series, exposure=possessions)
        dist2 = model.get_distribution(t2_prediction_series, exposure=possessions)

        chances_df.at[team2, team1] = get_victory_chance(dist1, dist2)
        chances_df.at[team1, team2] = get_victory_chance(dist2, dist1)

    return chances_df

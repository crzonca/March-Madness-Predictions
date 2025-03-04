import pandas as pd
import json


def map_full_names_to_school(config):
    with open(config.get('resource_locations').get('name_to_school'), 'r') as f:
        school_mapping = json.load(f)
    return school_mapping


def map_schools_to_full_name(config):
    name_mapping = {v: k for k, v in map_full_names_to_school(config).items()}
    return name_mapping


def conference_mapping(config):
    confs_df = pd.read_csv(config.get('resource_locations').get('true_records'))
    mapping = {row['team']: row['conference'] for index, row in confs_df.iterrows()}

    return mapping

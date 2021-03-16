from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from typing import List
from argparse import ArgumentParser
from src.utils import data_args

AUTHOR_SEP = ':::'
AUTHOR_ID = 'author_id'
TARGET = 'target'
POSTS = 'posts'


def convert_dataframe(data_path: str, lang: str):
    '''
    This method takes the path of the dataset as input and then convert it to dataframe for exploration
    :param data_path:
           lang: en or es
    :return: pd.Dataframe
    '''

    data_path = Path(data_path)

    profiles_path = data_path / lang
    labels_path = profiles_path / 'truth.txt'

    ground_truth = {}
    with open(labels_path, 'r') as r:
        labels = r.readlines()
        for label in labels:
            label = label.split(AUTHOR_SEP)
            ground_truth[label[0]] = int(label[1])

    profiles = []
    for profile_path in profiles_path.glob('*.xml'):
        tree = ET.parse(profile_path)
        root = tree.getroot()
        for child in root:
            posts = []
            for ch in child:
                posts.append(ch.text)

        author_id = profile_path.stem
        profiles.append(
            {
                AUTHOR_ID: author_id,
                POSTS: posts,
                TARGET: ground_truth[author_id]
            }
        )

    return pd.DataFrame(profiles)


def retrieve_joined_texts_targets(data):
    targets = data[TARGET].to_numpy()
    ids = data[AUTHOR_ID].to_numpy()

    profiles = data.groupby([TARGET, AUTHOR_ID])
    joined_posts = []
    for _, group in profiles:
        posts = group[POSTS].to_numpy()[0]
        joined_posts.append(' '.join(posts))

    assert len(ids) == len(targets) == len(joined_posts)
    return (ids, joined_posts, targets)


def prepare_for_pan_baseline(data_path, lang):
    df = convert_dataframe(data_path=data_path, lang=lang)
    ids, inputs, targets = retrieve_joined_texts_targets(df)
    return {
        'ids': ids,
        'inputs': inputs,
        'targets': targets

    }


@dataclass
class DataStats:
    post_len: List = field(default_factory=lambda: [])


def eda(data_path, lang):
    '''
    Data Exploration
    :param data_path:
    :return:
    '''
    data = convert_dataframe(data_path, lang)

    print(f'Class Distribution of {lang}')
    print(data.groupby([TARGET]).count())

    profiles = data.groupby([TARGET, AUTHOR_ID])

    hatespeech_profiles = DataStats()
    normal_profiles = DataStats()
    for (target, author_id), group in profiles:
        print(f'Requesting {author_id}')
        posts = group[POSTS].to_numpy()[0]
        print(f'Number of posts \n{len(posts)}')
        for post in posts:
            if target == 0:
                normal_profiles.post_len.append(len(post))
            else:
                hatespeech_profiles.post_len.append(len(post))
    print(
        f'Hatespeech spreaders use mean of {np.mean(hatespeech_profiles.post_len)} std {np.std(hatespeech_profiles.post_len)}')
    print(f'Normal profiles use mean of {np.mean(normal_profiles.post_len)} std {np.std(normal_profiles.post_len)}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = data_args(parser)
    args = parser.parse_args()
    eda(data_path=args.data, lang=args.lang)

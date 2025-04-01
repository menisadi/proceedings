# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(12,5)})
# %load_ext autoreload
# %autoreload 2

# %%
import re
from fuzzywuzzy import process
import pprint

# %%
df = pd.read_parquet("proceedings.parquet")

# %%
df["volume"] = df["conference"].apply(lambda x: x.split(":")[0].split(" ")[1])

# %%
df['year'] = df['conference'].apply(lambda x: re.search(r'(19|20)\d{2}', str(x)).group(0) if re.search(r'(19|20)\d{2}', str(x)) else None)

# %%
df['year'].value_counts()

# %%
df["conference_main"] = df["conference"].apply(lambda x: x.split(":")[1].split(',')[0])

# %%
df.loc[df['year'].isna(), 'conference'].unique()

# %%
df.loc[df['volume'] == '117', 'year'] = '2020'

# %%
df.loc[df['volume'] == '201', 'year'] = '2023'

# %%
df.loc[df['volume'] == '209', 'year'] = '2023'


# %%
def get_best_match(name, choices):
    return process.extractOne(name, choices)


# %%
df['conference_main'] = df['conference_main'].apply(lambda c: c[1:])

# %%
# df['conference_main'].apply(remove_prefix)

# %%
import networkx as nx

# %%
unique_conferences = set(df['conference_main'])

name_mapping = {}
names_graph = nx.Graph()
names_graph.add_nodes_from(unique_conferences)

for conference in unique_conferences:
    match, score = get_best_match(conference, unique_conferences - {conference})
    if score >= 90:
        names_graph.add_edge(conference, match)
        # name_mapping[conference] = match

# %%
for c in nx.connected_components(names_graph):
    if len(c) > 1:
        target_name = min(c, key=len)
        for conference in c:
            name_mapping[conference] = target_name

# %%
for c1, c2 in name_mapping.items():
    if 'learning theory' in c1.lower():
        print(f'{c1}\n -> {c2}\n')


# %%
# def remove_prefix(name):
#     pattern = r'.*?(\b(?!(with|health)\b)\w*th)\b.*?\s'
#     name = re.sub(pattern, '', name).strip()

#     return name

# %%
def clean_conference_name(name):
    pattern_prefix = r'.*?(\b(?!(with|health)\b)\w*th)\b.*?\s'
    name = re.sub(pattern_prefix, '', name).strip()
    name = re.sub(r'[0-9\']', '', name)

    return name


# %%
df['conference_main'] = df['conference_main'].map(name_mapping).fillna(df['conference_main']).apply(clean_conference_name)

# %%
[c for c in df['conference_main'].unique() if 'learning theory' in c.lower()]

# %%
(df[['conference_main', 'volume']].drop_duplicates()['conference_main'].value_counts() > 1).value_counts(normalize=True)

# %%
df.loc[df['year'].apply(int) > 2018, 'conference_main'].value_counts(normalize=False).tail(10)

# %%
df[['title', 'authors', 'conference_main', 'year', 'volume']]

# %%
df["authors_list"] = df["authors"].apply(lambda auths: auths.split(",\xa0"))

# %%
from collections import Counter

# %%
authors_counter = Counter([a for a_list in df["authors_list"].to_list() for a in a_list])

# %%
pd.DataFrame(authors_counter.most_common(10))

# %%
pd.Series(" ".join(df["title"].values.flatten()).lower().split(" ")).value_counts().head(10)

# %%
from nltk.corpus import stopwords 
from nltk.tokenize import wordpunct_tokenize

# %%
all_titles = " ".join(df["title"].values.flatten()).lower().split(" ")

# %%
all_titles_filtered = [w for w in all_titles if w not in stopwords.words("english")]

# %%
pd.Series(all_titles_filtered).value_counts(normalize=False).head(20)

# %%
is_it_causal = df["title"].apply(lambda title: 'caus' in title.lower())
most_published_causal = df.loc[is_it_causal, 'authors_list'].explode().value_counts().head(15)

# %%
most_published_causal

# %%
most_published_causal.plot.bar()
plt.xticks(rotation=45)
plt.title("Causal")

# %%
is_it_privacy = df["title"].apply(lambda title: 'priva' in title.lower())
most_published_privacy = df.loc[is_it_privacy, 'authors_list'].explode().value_counts().head(15)

# %%
most_published_privacy.plot.bar()
plt.xticks(rotation=45)
plt.title("Privacy")

# %%
df['conference_main'].value_counts().head(10)

# %%
is_it_active = df["title"].apply(lambda title: 'activ' in title.lower())
most_published_active = df.loc[is_it_active, 'authors_list'].explode().value_counts().head(15)

# %%
most_published_active.plot.bar()
plt.xticks(rotation=45)
plt.title("Active")

# %%
is_it_reinforcement = df["title"].apply(lambda title: 'reinforcement' in title.lower())
most_published_reinforcement = df.loc[is_it_reinforcement, 'authors_list'].explode().value_counts().head(15)

# %%
most_published_reinforcement.plot.bar()
plt.xticks(rotation=45)
plt.title("Reinforcement")

# %%

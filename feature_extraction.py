import pandas as pd
import re
import networkx as nx
from fuzzywuzzy import process
from nltk.corpus import stopwords


def get_best_match(name, choices):
    return process.extractOne(name, choices)


def clean_conference_name(name):
    pattern_prefix = r".*?(\b(?!(with|health)\b)\w*th)\b.*?\s"
    name = re.sub(pattern_prefix, "", name).strip()
    name = re.sub(r"[0-9\']", "", name)

    return name


df = pd.read_parquet("proceedings.parquet")
df["volume"] = df["conference"].apply(lambda x: x.split(":")[0].split(" ")[1])
df["year"] = df["conference"].apply(
    lambda x: re.search(r"(19|20)\d{2}", str(x)).group(0)
    if re.search(r"(19|20)\d{2}", str(x))
    else None
)
df["conference_main"] = df["conference"].apply(
    lambda x: x.split(":")[1].split(",")[0]
)
df.loc[df["volume"] == "117", "year"] = "2020"
df.loc[df["volume"] == "201", "year"] = "2023"
df.loc[df["volume"] == "209", "year"] = "2023"

unique_conferences = set(df["conference_main"])

name_mapping = {}
names_graph = nx.Graph()
names_graph.add_nodes_from(unique_conferences)

for conference in unique_conferences:
    match, score = get_best_match(
        conference, unique_conferences - {conference}
    )
    if score >= 90:
        names_graph.add_edge(conference, match)

for c in nx.connected_components(names_graph):
    if len(c) > 1:
        target_name = min(c, key=len)
        for conference in c:
            name_mapping[conference] = target_name

df["conference_main"] = (
    df["conference_main"]
    .map(name_mapping)
    .fillna(df["conference_main"])
    .apply(clean_conference_name)
)

df["authors_list"] = df["authors"].apply(lambda auths: auths.split(",\xa0"))

all_titles = " ".join(df["title"].values.flatten()).lower().split(" ")
all_titles_filtered = [
    w for w in all_titles if w not in stopwords.words("english")
]
most_common_words = (
    pd.Series(all_titles_filtered).value_counts(normalize=False).head(20)
)

key_word = "caus"
does_it_include_the_word = df["title"].apply(
    lambda title: key_word in title.lower()
)
most_published_within_this_word = (
    df.loc[does_it_include_the_word, "authors_list"]
    .explode()
    .value_counts()
    .head(15)
)

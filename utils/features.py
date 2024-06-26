# utils/features.py
import re
from fuzzywuzzy import process
import pandas as pd
import networkx as nx


def get_best_match(name: str, choices: list[str]) -> tuple:
    """
    Get the best match for a name from a list of choices.

    Args:
    name (str): The name to find a match for.
    choices (list[str]): A list of possible matches.

    Returns:
    tuple: A tuple containing the best match and its score.
    """
    return process.extractOne(name, choices)


def clean_conference_name(name: str) -> str:
    """
    Clean a conference name by removing unwanted characters.

    Args:
    name (str): The conference name to clean.

    Returns:
    str: The cleaned conference name.
    """
    pattern_prefix = r".*?(\b(?!(with|health)\b)\w*th)\b.*?\s"
    name = re.sub(pattern_prefix, "", name).strip()
    name = re.sub(r"[0-9\']", "", name)

    return name


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic features to a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to add features to.

    Returns:
    pd.DataFrame: The DataFrame with added features.
    """
    df["volume"] = df["conference"].apply(
        lambda x: x.split(":")[0].split(" ")[1]
    )
    df["year"] = df["conference"].apply(
        lambda x: re.search(r"(19|20)\d{2}", str(x)).group(0)
        if re.search(r"(19|20)\d{2}", str(x))
        else None
    )
    df.loc[df["volume"] == "117", "year"] = "2020"
    df.loc[df["volume"] == "201", "year"] = "2023"
    df.loc[df["volume"] == "209", "year"] = "2023"

    df["authors_list"] = df["authors"].apply(
        lambda auths: auths.split(",\xa0")
    )

    return df


def add_clean_conference_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clean conference names to a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to add clean conference names to.

    Returns:
    pd.DataFrame: The DataFrame with clean conference names.
    """
    df["conference_main"] = df["conference"].apply(
        lambda x: x.split(":")[1].split(",")[0]
    )
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

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features to a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to add features to.

    Returns:
    pd.DataFrame: The DataFrame with added features.
    """
    df = add_basic_features(df)
    df = add_clean_conference_names(df)
    return df

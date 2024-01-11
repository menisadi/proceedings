# utils/tables.py
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords 

def create_authors_publications_list(df, top_n=10):
    """
    Create a DataFrame that lists the authors and their respective publications.
    """
    authors_list = df["authors_list"].explode()
    authors_publications = authors_list.value_counts().reset_index()
    authors_publications.columns = ["Author", "Publications"]
    return authors_publications.head(top_n)


def top_authors_by_keyword(df, keyword, top_n=10):
    """
    Get the top authors based on the number of publications containing a specific keyword.
    """
    keyword_in_title = df["title"].apply(
        lambda title: keyword in title.lower()
    )
    keyword_authors = (
        df.loc[keyword_in_title, "authors_list"]
        .explode()
        .value_counts()
        .reset_index()
    )
    keyword_authors.columns = ["Author", "Publications"]
    return keyword_authors.head(top_n)


def most_common_keywords(df, top_n=10):
    """
    Get the most common keywords in the titles of the publications.
    """
    all_titles = " ".join(df["title"].values.flatten()).lower().split(" ")
    all_titles_filtered = [
        w for w in all_titles if w not in stopwords.words("english")
    ]
    most_common_words = (
        pd.Series(all_titles_filtered)
        .value_counts(normalize=False)
    )
    return most_common_words.head(top_n)

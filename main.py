# main.py
import argparse
import pandas as pd
from utils.extract import extract_paper_data, create_dataframe
from utils.features import add_features
from utils.tables import (
    create_authors_publications_list,
    top_authors_by_keyword,
    most_common_keywords,
)


def main(args):
    # List of conference URLs
    if args.custom_conference_urls:
        conference_urls = args.custom_conference_urls
    else:
        conference_urls = [
            "https://proceedings.mlr.press/v" + str(i)
            for i in range(1, args.default_conference_count)
        ]

    # Create DataFrame
    df = create_dataframe(conference_urls)
    print("Dataframe created")
    print(f"Listing {len(df)} papers")

    # Add features
    df = add_features(df)

    # Save DataFrame based on user preference
    if args.save_csv:
        df.to_csv("proceedings.csv", sep="\t")
        print("DataFrame saved as CSV.")
    elif args.save_parquet:
        df.to_parquet("proceedings.parquet")
        print("DataFrame saved as Parquet.")

    # Create induced dataframes and tables based on user preferences
    if args.generate_common_words:
        common_keywords = most_common_keywords(df)
        print("\nMost Common Keywords:")
        print(common_keywords)

    if args.generate_authors_list:
        authors_publications = create_authors_publications_list(df)
        print("\nAuthors and their Publications:")
        print(authors_publications)

    if args.generate_authors_by_keyword:
        top_authors = top_authors_by_keyword(df, args.keyword)
        print(f"\nTop Authors with Publications containing '{args.keyword}':")
        print(top_authors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process conference data.")
    parser.add_argument(
        "-c", "--save-csv", action="store_true", help="Save DataFrame as CSV."
    )
    parser.add_argument(
        "-p",
        "--save-parquet",
        action="store_true",
        help="Save DataFrame as Parquet.",
    )
    parser.add_argument(
        "-g",
        "--generate-common-words",
        action="store_true",
        help="Generate most common words table.",
    )
    parser.add_argument(
        "-a",
        "--generate-authors-list",
        action="store_true",
        help="Generate authors and their publications table.",
    )
    parser.add_argument(
        "-k",
        "--generate-authors-by-keyword",
        action="store_true",
        help="Generate top authors by keyword table.",
    )
    parser.add_argument(
        "-w",
        "--keyword",
        type=str,
        help="Keyword for authors by keyword analysis.",
    )
    parser.add_argument(
        "-u",
        "--custom-conference-urls",
        nargs="+",
        type=str,
        help="Custom list of conference URLs.",
    )
    parser.add_argument(
        "-d",
        "--default-conference-count",
        type=int,
        default=222,
        help="Default number of conferences to consider.",
    )

    args = parser.parse_args()
    main(args)

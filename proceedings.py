from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to extract paper data from a conference page
def extract_paper_data(conference_url):
    response = requests.get(conference_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract conference information
    conference_name_elem = soup.find('h2')
    conference_name = conference_name_elem.text.strip() if conference_name_elem else ''
    conference_number = conference_url.split('/')[-1]

    # Extract paper information
    papers = []
    for paper_div in soup.find_all('div', class_='paper'):
        title = paper_div.find('p', class_='title').text.strip()

        # Check if 'authors' span element exists within 'details'
        details_elem = paper_div.find('p', class_='details')
        authors_elem = details_elem.find('span', class_='authors') if details_elem else None
        authors = authors_elem.text.strip() if authors_elem else ''

        papers.append({'title': title, 'authors': authors, 'conference': conference_name})

    return papers

# Function to create a DataFrame from the extracted data
def create_dataframe(conference_urls):
    all_papers = []
    for conference_url in tqdm(conference_urls):
        papers = extract_paper_data(conference_url)
        all_papers.extend(papers)

    df = pd.DataFrame(all_papers)
    return df

# List of conference URLs
conference_urls = ['https://proceedings.mlr.press/v' + str(i) for i in range(1, 222)]

# Create DataFrame
df = create_dataframe(conference_urls)
print("Dataframe created")
print(f"Listing {len(df)} papers")

# Create the DataFrame
df.to_csv("proceedings.csv", sep='\t')
df.to_parquet("proceedings.parquet")
print("Done!")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin
import re

BASE_URL = "https://www.tapology.com"
MAIN_URL = f"{BASE_URL}/rankings/groups/annual"
OUTPUT_PATH = "../data/raw/ufc_fights.csv"

def fetch_page(url):
    """Fetch the webpage content."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch {url} (Status code: {response.status_code})")
        return None

def fetch_year_links(base_url):
    """Fetch year links dynamically from the main page."""
    main_page_content = fetch_page(base_url)
    if not main_page_content:
        return []
    return extract_year_links(main_page_content)

def extract_year_links(main_page_content):
    """Extract links to yearly fight rankings dynamically."""
    soup = BeautifulSoup(main_page_content, "html.parser")
    year_links = []

    # Locate ranking groups
    ranking_groups = soup.find_all("div", class_="rankingGroup")

    for group in ranking_groups:
        # Look for links within the ranking group
        link_tag = group.find("h2", class_="consensus").find("a", href=True)
        if link_tag:
            href = link_tag["href"]
            # Ensure the link contains the year pattern and is not redundant
            if href and "fights-of-the-year" in href:
                full_url = urljoin(BASE_URL, href)
                year_links.append(full_url)

    # Remove duplicates, if any
    year_links = list(set(year_links))
    print(f"Extracted {len(year_links)} year links: {year_links}")
    return year_links

def extract_pagination_links(year_page_content, base_url):
    """Extract pagination links for the annual rankings."""
    soup = BeautifulSoup(year_page_content, "html.parser")
    pagination_links = [base_url]
    pagination_nav = soup.find("nav", class_="pagination")
    if pagination_nav:
        for link in pagination_nav.find_all("a", href=True):
            full_link = urljoin(base_url, link["href"])
            if full_link not in pagination_links:
                pagination_links.append(full_link)
    return pagination_links

def extract_year_from_url(url):
    """
    Extract the correct year from the URL based on specific patterns.
    
    Handles two main URL formats:
    1. https://www.tapology.com/rankings/2010-best-mma-and-ufc-fights-of-the-year
       Extract year from the first part before the hyphen
    
    2. https://www.tapology.com/rankings/1351-2014-mma-fights-of-the-year
       Extract year from the second part after the first hyphen
    """
    # First, try to match the pattern for URLs like 1351-2014-mma-fights-of-the-year
    pattern1 = r'/rankings/\d+-(\d{4})-'
    match1 = re.search(pattern1, url)
    if match1:
        return match1.group(1)
    
    # Then, try to match the pattern for URLs like 2010-best-mma-and-ufc-fights-of-the-year
    pattern2 = r'/rankings/(\d{4})-'
    match2 = re.search(pattern2, url)
    if match2:
        return match2.group(1)
    
    return None

def extract_ufc_fights(year_page_content, year):
    """Extract UFC fights from a yearly rankings page."""
    soup = BeautifulSoup(year_page_content, "html.parser")
    fights = []

    for fight in soup.find_all("li", class_="rankingItemsItem"):
        try:
            rank = fight.find("p", class_="rankingItemsItemRank").get_text(strip=True)
            name = fight.find("a").get_text(strip=True)
            event = fight.find("h1", class_="right").get_text(strip=True)
            date = fight.find("span", class_="right").get_text(strip=True)
            result = fight.find("p").get_text(strip=True).split(" via ")[-1]
            is_ufc = "UFC" in event
        except AttributeError:
            # Skip fights with missing data
            continue

        fights.append({
            "Year": year,
            "Rank": rank,
            "Fight": name,
            "Event": event,
            "Date": date,
            "Result": result,
            "Is UFC": is_ufc
        })
    return fights

def save_to_csv(data, file_path):
    """Save the fight data to a CSV file."""
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directories if needed
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    # Step 1: Fetch yearly links
    year_links = fetch_year_links(MAIN_URL)
    print(f"Found {len(year_links)} yearly links.")

    all_fights = []

    # Step 2: Process each yearly page
    for year_url in year_links:
        print(f"Processing {year_url}...")
        
        # Extract year from URL
        year = extract_year_from_url(year_url)
        
        if not year:
            print(f"Could not extract year from {year_url}. Skipping.")
            continue

        year_page_content = fetch_page(year_url)

        if not year_page_content:
            print(f"Skipping {year_url} due to fetch failure.")
            continue

        # Handle pagination
        pagination_links = extract_pagination_links(year_page_content, year_url)
        for page_url in pagination_links:
            print(f"Processing page: {page_url}")
            page_content = fetch_page(page_url)
            if page_content:
                ufc_fights = extract_ufc_fights(page_content, year)
                all_fights.extend(ufc_fights)
                print(f"Extracted {len(ufc_fights)} fights from {page_url}")

    # Step 3: Save to CSV
    if all_fights:
        save_to_csv(all_fights, OUTPUT_PATH)
    else:
        print("No fights were scraped.")
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import concurrent.futures

# Set project root dynamically to the correct path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(
    project_root, 
    'data', 
    'raw', 
    'ufc_fighters_roster_stats_and_records.csv'
)

BASE_URL = "https://www.ufc.com"
ATHLETES_URL = f"{BASE_URL}/athletes/all"

def fetch_page(url):
    """Fetch webpage content."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch {url} (Status code: {response.status_code})")
        return None

def get_athletes():
    """Scrape all athletes from the UFC athletes page."""
    athletes = []
    page = 0

    while True:
        url = f"{ATHLETES_URL}?page={page}"
        print(f"Fetching page: {url}")
        page_content = fetch_page(url)
        if not page_content:
            break

        soup = BeautifulSoup(page_content, "html.parser")
        fighter_cards = soup.find_all("div", class_="c-listing-athlete-flipcard__inner")

        if not fighter_cards:
            print("No more fighters found.")
            break

        for card in fighter_cards:
            name_tag = card.find("span", class_="c-listing-athlete__name")
            if name_tag:
                name = name_tag.get_text(strip=True)
                athletes.append({"name": name})

        # Check for "Load More" button
        load_more = soup.find("a", class_="button", title="Load more items", rel="next")
        if not load_more:
            break

        page += 1
        time.sleep(1)  # Avoid hammering the server

    return athletes

def parse_stat(soup, label, tag, class_name):
    """Helper function to parse a stat."""
    element = soup.find(tag, class_=class_name)
    return element.get_text(strip=True) if element else None

def parse_echart_stat(soup, label):
    """Helper function to parse stats from e-chart-circle elements."""
    elements = soup.find_all("title")
    for element in elements:
        if label in element.get_text():
            return element.get_text(strip=True).replace(label, "").strip()
    return None

def parse_stat_group(soup, label):
    """Helper function to parse stat groups."""
    groups = soup.find_all("div", class_="c-stat-3bar__group")
    stats = {}
    for group in groups:
        group_label = group.find("div", class_="c-stat-3bar__label")
        group_value = group.find("div", class_="c-stat-3bar__value")
        if group_label and group_value:
            stats[group_label.get_text(strip=True)] = group_value.get_text(strip=True)
    return stats.get(label, None)

def get_fighter_stats(fighter_name):
    """Fetch detailed stats for a single fighter."""
    name_parts = fighter_name.lower().split()
    profile_url = f"{BASE_URL}/athlete/{'-'.join(name_parts)}"
    print(f"Fetching stats for: {fighter_name} from {profile_url}")

    page_content = fetch_page(profile_url)
    if not page_content:
        return None

    soup = BeautifulSoup(page_content, "html.parser")
    stats = {}

    # Fetch Fighter's Name and Nickname
    stats['name'] = parse_stat(soup, "Name", "h1", "hero-profile__name")
    nickname_tag = soup.find("p", class_="hero-profile__nickname")
    stats['nickname'] = nickname_tag.get_text(strip=True).strip('"') if nickname_tag else None

    # Fetch Division and Record
    stats['division'] = parse_stat(soup, "Division", "p", "hero-profile__division-title")
    stats['record'] = parse_stat(soup, "Record", "p", "hero-profile__division-body")

    # Additional Bio Information
    bio_fields = soup.find_all("div", class_="c-bio__field")
    for field in bio_fields:
        label = field.find("div", class_="c-bio__label")
        value = field.find("div", class_="c-bio__text")
        if label and value:
            label_text = label.get_text(strip=True)
            value_text = value.get_text(strip=True)
            
            if label_text == "Status":
                stats['status'] = value_text
            elif label_text == "Place of Birth":
                stats['place_of_birth'] = value_text
            elif label_text == "Age":
                stats['age'] = value_text
            elif label_text == "Height":
                stats['height'] = value_text
            elif label_text == "Weight":
                stats['weight'] = value_text
            elif label_text == "Reach":
                stats['reach'] = value_text
            elif label_text == "Leg reach":
                stats['leg_reach'] = value_text

    # Fetch striking and takedown accuracy from e-chart-circle
    stats['striking_accuracy'] = parse_echart_stat(soup, "Striking accuracy")
    stats['takedown_accuracy'] = parse_echart_stat(soup, "Takedown Accuracy")

    # Fetch other comparative stats
    stats['sig_str_defense'] = parse_stat(soup, "Sig. Str. Defense", "div", "c-stat-compare__number")
    stats['takedown_defense'] = parse_stat(soup, "Takedown Defense", "div", "c-stat-compare__number")
    stats['knockdown_avg'] = parse_stat(soup, "Knockdown Avg", "div", "c-stat-compare__number")
    stats['average_fight_time'] = parse_stat(soup, "Average Fight Time", "div", "c-stat-compare__number")

    # Fetch Significant Strikes by Position
    sig_strikes_pos_group = soup.find("div", class_="c-stat-3bar__legend")
    if sig_strikes_pos_group:
        pos_groups = sig_strikes_pos_group.find_all("div", class_="c-stat-3bar__group")
        pos_stats = {}
        for group in pos_groups:
            label = group.find("div", class_="c-stat-3bar__label")
            value = group.find("div", class_="c-stat-3bar__value")
            if label and value:
                pos_stats[label.get_text(strip=True).lower()] = value.get_text(strip=True)
        
        stats['sig_strikes_standing'] = pos_stats.get('standing', None)
        stats['sig_strikes_clinch'] = pos_stats.get('clinch', None)
        stats['sig_strikes_ground'] = pos_stats.get('ground', None)

    # Fetch Significant Strikes by Target
    sig_strikes_target = soup.find_all("g", id=lambda x: x and x.startswith("e-stat-body_"))
    sig_strikes_by_target = {}
    for target in sig_strikes_target:
        target_name = target.find("text", text=lambda x: x in ["Head", "Body", "Leg"])
        if target_name:
            target_text = target_name.get_text(strip=True)
            value_tag = target.find("text", id=lambda x: "_value" in str(x))
            percent_tag = target.find("text", id=lambda x: "_percent" in str(x))
            
            if value_tag and percent_tag:
                sig_strikes_by_target[target_text.lower()] = {
                    'value': value_tag.get_text(strip=True),
                    'percent': percent_tag.get_text(strip=True)
                }

    stats['sig_strikes_head'] = f"{sig_strikes_by_target.get('head', {}).get('value', '')} ({sig_strikes_by_target.get('head', {}).get('percent', '')})" if 'head' in sig_strikes_by_target else None
    stats['sig_strikes_body'] = f"{sig_strikes_by_target.get('body', {}).get('value', '')} ({sig_strikes_by_target.get('body', {}).get('percent', '')})" if 'body' in sig_strikes_by_target else None
    stats['sig_strikes_leg'] = f"{sig_strikes_by_target.get('leg', {}).get('value', '')} ({sig_strikes_by_target.get('leg', {}).get('percent', '')})" if 'leg' in sig_strikes_by_target else None

    # Fetch Win By Method
    win_by_method_groups = soup.find_all("div", class_="c-stat-3bar__group")
    win_by_method = {}
    for group in win_by_method_groups:
        label_tag = group.find("div", class_="c-stat-3bar__label")
        value_tag = group.find("div", class_="c-stat-3bar__value")
        if label_tag and value_tag:
            win_by_method[label_tag.get_text(strip=True)] = value_tag.get_text(strip=True)

    stats['wins_by_ko_tkо'] = win_by_method.get('KO/TKO', None)
    stats['wins_by_dec'] = win_by_method.get('DEC', None)
    stats['wins_by_sub'] = win_by_method.get('SUB', None)

    # Fetch original stats
    stats['sig_strikes_landed'] = parse_stat(soup, "Sig. Strikes Landed", "dd", "c-overlap__stats-value")
    stats['sig_strikes_attempted'] = parse_stat(soup, "Sig. Strikes Attempted", "dd", "c-overlap__stats-value")
    stats['takedowns_landed'] = parse_stat(soup, "Takedowns Landed", "dd", "c-overlap__stats-value")
    stats['takedowns_attempted'] = parse_stat(soup, "Takedowns Attempted", "dd", "c-overlap__stats-value")

    # Compare stats
    stats['sig_str_landed_per_min'] = parse_stat(soup, "Sig. Str. Landed per Min", "div", "c-stat-compare__number")
    stats['sig_str_absorbed_per_min'] = parse_stat(soup, "Sig. Str. Absorbed per Min", "div", "c-stat-compare__number")
    stats['takedown_avg_per_15_min'] = parse_stat(soup, "Takedown Avg Per 15 Min", "div", "c-stat-compare__number")
    stats['submission_avg_per_15_min'] = parse_stat(soup, "Submission Avg per 15 Min", "div", "c-stat-compare__number")

    return stats

def fetch_all_fighter_stats(athletes):
    """Fetch stats for all fighters concurrently."""
    all_fighter_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_fighter = {executor.submit(get_fighter_stats, athlete["name"]): athlete for athlete in athletes}
        for future in concurrent.futures.as_completed(future_to_fighter):
            athlete = future_to_fighter[future]
            try:
                stats = future.result()
                if stats:
                    all_fighter_data.append(stats)
            except Exception as e:
                print(f"Error fetching stats for {athlete['name']}: {e}")
    return all_fighter_data

def main():
    start_time = time.time()  # Start the timer

    # Step 1: Get all athletes
    print("Scraping all UFC athletes...")
    athletes = get_athletes()
    print(f"Found {len(athletes)} athletes.")

    # Step 2: Fetch stats for each athlete concurrently
    all_fighter_data = fetch_all_fighter_stats(athletes)

    # Step 3: Save to CSV
    print(f"Saving data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(all_fighter_data)
    df.to_csv(output_path, index=False)
    print("Data scraping complete!")

    # End the timer and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
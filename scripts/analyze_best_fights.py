import os
import pandas as pd

# File paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ufc_fights_path = os.path.join(project_root, 'UFCEventPredictionModel', 'data', 'raw', 'ufc_fights.csv')
large_dataset_path = os.path.join(project_root, 'UFCEventPredictionModel', 'data', 'raw', 'ufc_events_and__fights_stats.csv')
merged_output_path = os.path.join(project_root, 'UFCEventPredictionModel', 'data', 'processed', 'merged_best_fights.csv')
intersecting_output_path = os.path.join(project_root, 'UFCEventPredictionModel', 'data', 'processed', 'intersecting_datapoints.csv')
missing_names_output_path = os.path.join(project_root, 'UFCEventPredictionModel', 'data', 'processed', 'missing_name_in_large_dataset.csv')

# Load datasets
ufc_fights = pd.read_csv(ufc_fights_path)
large_dataset = pd.read_csv(large_dataset_path)

# Filter rows in ufc_fights where isUFC is True
ufc_fights = ufc_fights[ufc_fights['Is UFC'] == True]

# Normalize event names in both datasets (optional, if you want to keep this for any reason)
def normalize_event(event_name):
    if "UFC Fight Night" in event_name:
        return "UFC Fight Night"
    elif "UFC" in event_name:
        return event_name.split(':')[0].strip()  # Keep only "UFC <number>"
    return event_name

ufc_fights['Event'] = ufc_fights['Event'].apply(normalize_event)
large_dataset['event_name'] = large_dataset['event_name'].apply(normalize_event)

# Split fighter names in ufc_fights
ufc_fights[['fighter_a', 'fighter_b']] = ufc_fights['Fight'].str.split(' vs. ', expand=True)

# Strip whitespace and convert to lowercase
ufc_fights['fighter_a'] = ufc_fights['fighter_a'].str.strip().str.lower()
ufc_fights['fighter_b'] = ufc_fights['fighter_b'].str.strip().str.lower()
large_dataset['r_fighter'] = large_dataset['r_fighter'].str.strip().str.lower()
large_dataset['b_fighter'] = large_dataset['b_fighter'].str.strip().str.lower()

# Initialize results
merged_best_fights = []
intersecting_datapoints = []
missing_names = []

# Iterate over rows in ufc_fights
for _, row in ufc_fights.iterrows():
    fighter_a = row['fighter_a'].split()[-1]  # Extract last name
    fighter_b = row['fighter_b'].split()[-1]  # Extract last name

    # Filter large_dataset for matching fighter names (ignoring event name)
    matches = large_dataset[
        ((large_dataset['r_fighter'].str.contains(fighter_a)) & 
         (large_dataset['b_fighter'].str.contains(fighter_b))) |
        ((large_dataset['r_fighter'].str.contains(fighter_b)) & 
         (large_dataset['b_fighter'].str.contains(fighter_a)))
    ]

    # Handle multiple matches
    if len(matches) > 1:
        intersecting_datapoints.append(matches)
    elif len(matches) == 1:
        merged_best_fights.append(matches.iloc[0])
    else:
        # If no matches found, add the row to missing_names list
        missing_names.append(row)

# Combine results into DataFrames
merged_best_fights_df = pd.DataFrame(merged_best_fights)
intersecting_datapoints_df = pd.concat(intersecting_datapoints, ignore_index=True)
missing_names_df = pd.DataFrame(missing_names)

# Save results to CSV
merged_best_fights_df.to_csv(merged_output_path, index=False)
intersecting_datapoints_df.to_csv(intersecting_output_path, index=False)

# Save missing fighter names to a separate CSV
missing_names_df.to_csv(missing_names_output_path, index=False)

print(f"Merged fights saved to {merged_output_path}")
print(f"Intersecting fights saved to {intersecting_output_path}")
print(f"Missing fighter names saved to {missing_names_output_path}")
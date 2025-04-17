import pandas as pd

# Step 1: Preprocessing the transcript data
# Import the transcript CSV file as a pandas DataFrame
transcript_df = pd.read_csv('1_10_seasons_tbbt.csv')

# Extract season and episode numbers from the 'episode_name' column
def extract_season_episode(row):
    try:
        season = int(row.split(" ")[1])  # Extract the season number
        episode = int(row.split(" ")[3])  # Extract the episode number
        return season, episode
    except (IndexError, ValueError):
        return None, None  # Handle rows that don't match the expected format

# Apply the function to extract season and episode numbers
transcript_df[['Season', 'Episode']] = transcript_df['episode_name'].apply(
    lambda x: pd.Series(extract_season_episode(x))
)

# Drop rows with invalid season or episode numbers
transcript_df = transcript_df.dropna(subset=['Season', 'Episode'])

# Convert Season and Episode columns to integers
transcript_df['Season'] = transcript_df['Season'].astype(int)
transcript_df['Episode'] = transcript_df['Episode'].astype(int)

# Calculate the overall episode number
# First, sort the DataFrame by Season and Episode
transcript_df = transcript_df.sort_values(['Season', 'Episode'])

# Create a mapping of (Season, Episode) to overall episode number
episode_mapping = {}
overall_episode = 1
for _, row in transcript_df[['Season', 'Episode']].drop_duplicates().iterrows():
    episode_mapping[(row['Season'], row['Episode'])] = overall_episode
    overall_episode += 1

# Map the (Season, Episode) pairs to the overall episode number
transcript_df['Episode_Number'] = transcript_df.apply(
    lambda row: episode_mapping[(row['Season'], row['Episode'])], axis=1
)

# Step 2: Filtering and analysis
# Define the list of valid characters
valid_characters = ['Leonard', 'Sheldon', 'Penny', 'Raj', 'Howard', 'Bernadette', 'Amy', 'Stuart']

# Filter the DataFrame to keep only rows where person_scene is in the valid_characters list
transcript_df = transcript_df[transcript_df['person_scene'].isin(valid_characters)]

# Group by Episode_Number and person_scene to count the number of lines per character per episode
line_counts = transcript_df.groupby(['Episode_Number', 'person_scene']).size().reset_index(name='Line_Count')

# Group by Episode_Number to calculate the total number of lines per episode
total_lines_per_episode = transcript_df.groupby('Episode_Number').size().reset_index(name='Total_Lines')

# Merge the line counts with the total lines per episode
line_counts = line_counts.merge(total_lines_per_episode, on='Episode_Number')

# Create a complete list of all combinations of Episode_Number and valid_characters
all_combinations = pd.MultiIndex.from_product(
    [line_counts['Episode_Number'].unique(), valid_characters],
    names=['Episode_Number', 'person_scene']
).to_frame(index=False)

# Merge the complete list with the actual line counts
line_counts = all_combinations.merge(line_counts, on=['Episode_Number', 'person_scene'], how='left')

# Fill missing values with 0 for Line_Count and Total_Lines
line_counts['Line_Count'] = line_counts['Line_Count'].fillna(0)
line_counts['Total_Lines'] = line_counts['Total_Lines'].fillna(0)

# Calculate the percentage of lines for each character per episode
line_counts['Line_Percentage'] = (line_counts['Line_Count'] / line_counts['Total_Lines'].replace(0, 1)) * 100

# Save the final results to a new CSV file
line_counts.to_csv('line_percentages_per_episode.csv', index=False)

# Display the first few rows of the final DataFrame to verify
print(line_counts.head())
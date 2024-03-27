import os
import subprocess
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Constants
DB_PATH = 'kilter_data.db'
LAYOUT_ID = 1
ASCENTS_THRESHOLD = 1
SIZE = (177, 185)


def initialize_database_if_needed(db_path=DB_PATH):
    """Check if the database exists and initialize it if it doesn't."""
    if not os.path.exists(db_path):
        print(f"Database at {db_path} not found. Initializing database...")
        # Run the command to initialize the database
        command = f"python3 -m boardlib database kilter {db_path}"
        subprocess.run(command, shell=True, check=True)
        print("Database initialized.")


def load_data_from_db():
    """Load and merge data from SQLite database."""
    sql = """
    SELECT placements.id, holes.x, holes.y
    FROM holes
    INNER JOIN placements ON placements.hole_id = holes.id
    WHERE placements.layout_id = ?
    """
    
    with sqlite3.connect(DB_PATH) as conn:
        holes_df = pd.read_sql_query(sql, conn, params=(LAYOUT_ID,))
        routes_df = pd.read_sql_query("SELECT * FROM climbs", conn)
        routes_grade_df = pd.read_sql_query("SELECT * FROM climb_stats", conn)
        grade_comparision_df = pd.read_sql_query("SELECT * FROM difficulty_grades", conn)

    
    return holes_df, routes_df, routes_grade_df, grade_comparision_df

def preprocess_data(routes_df, routes_grade_df, ascents = ASCENTS_THRESHOLD):
    """Filter and merge data based on ascents and layout."""
    routes_grade_df = routes_grade_df[routes_grade_df['ascensionist_count'] > ascents]
    routes_l1 = routes_df[routes_df['layout_id'] == 1][['uuid', 'frames']]
    diffs = routes_grade_df[['climb_uuid', 'angle', 'difficulty_average']]
    diffs = diffs.copy()
    diffs.rename(columns={"climb_uuid": "uuid"}, inplace=True)

    routes_l1 = routes_l1[~routes_l1['frames'].str.contains('x', na=False)]
    routes_l1 = routes_l1.merge(diffs, on='uuid', how='inner')

    return routes_l1

def get_features(routes_l1, holes_df, size=(177, 185)):
    """Generate feature matrix from route frames and hole positions."""
    # Initialize matrices to store route features and occupancy grid
    num_routes = len(routes_l1['frames'])
    routes_matrix = np.zeros((num_routes, *size))
    routes_occupied = np.zeros(size)
    
    # Preprocess hole positions for efficient lookup
    id_to_position = {row.id: (row.x + 20, row.y) for _, row in holes_df.iterrows()}

    for i, route_str in enumerate(routes_l1['frames']):
        # Clean and split route frame string into individual holds
        route_holds = route_str.replace(',', '').replace('"', '').split('p')
        route_holds = [hold for hold in route_holds if hold]  # Remove empty strings

        for hold_str in route_holds:
            try:
                # Split hold string into position and type, convert to integers
                pos, hold_type = map(int, hold_str.split('r'))
                if pos in id_to_position:  # Check if position is valid
                    x, y = id_to_position[pos]
                    # Map hold type to route matrix, adjust for indexing from bottom left
                    routes_matrix[i, size[0] - y, x] = hold_type
                    routes_occupied[size[0] - y, x] = 1
            except ValueError as e:
                print(f"Error parsing hold string '{hold_str}' in route {i}: {e}")

    # Identify non-empty rows and columns to reduce matrix size
    non_empty_rows = np.any(routes_occupied, axis=1)
    non_empty_cols = np.any(routes_occupied, axis=0)
    
    # Create a placeholder matrix for non-empty regions
    placeholder_matrix = np.zeros((num_routes, non_empty_rows.sum(), non_empty_cols.sum()))

    # Fill placeholder matrix with non-empty route data
    for i, route in enumerate(routes_matrix):
        placeholder_matrix[i] = route[non_empty_rows][:, non_empty_cols]

    return placeholder_matrix


class SimpleMLPRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 516)
        self.fc2 = nn.Linear(516, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RoutesDataset(Dataset):
    def __init__(self, features, labels, angle):
        self.features = features.reshape(features.shape[0], -1)
        self.labels = labels
        self.angle = angle.reshape(-1, 1)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(np.hstack([self.features, self.angle]))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.double), torch.tensor(self.labels[index], dtype=torch.double)

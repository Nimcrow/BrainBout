import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import os
import time
import logging
import joblib
from tqdm import tqdm

class UFCFightPreprocessor:
    def __init__(self, base_dir=None):
        """
        Initialize preprocessor with project base directory and set up logging
        
        Parameters:
        - base_dir: Base directory of the UFCEventPredictionModel project
        """
        self.start_time = time.time()
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(base_dir)  # Go up one level to project root
        
        events_path = os.path.join(base_dir, 'data', 'raw', 'ufc_events_and__fights_stats.csv')
        best_fights_path = os.path.join(base_dir, 'data', 'processed', 'Final_merged_fights.csv')
        roster_path = os.path.join(base_dir, 'data', 'raw', 'ufc_fighters_roster_stats_and_records.csv')
        
        logging.info(f"Starting to load datasets from: {base_dir}")
        start_read = time.time()
        logging.info(f"Loading events dataset from: {events_path}")
        self.events_df = pd.read_csv(events_path)
        logging.info(f"Events dataset loaded. Shape: {self.events_df.shape}")
        logging.info(f"Loading best fights dataset from: {best_fights_path}")
        self.best_fights_df = pd.read_csv(best_fights_path)
        logging.info(f"Best fights dataset loaded. Shape: {self.best_fights_df.shape}")
        logging.info(f"Loading roster dataset from: {roster_path}")
        self.roster_df = pd.read_csv(roster_path)
        logging.info(f"Roster dataset loaded. Shape: {self.roster_df.shape}")
        logging.info(f"Dataset loading completed in {time.time() - start_read:.2f} seconds")
        self.base_dir = base_dir
        self._create_watchability_target()
        
        self.base_dir = base_dir
        self._create_watchability_target()

    def _calculate_fight_metrics(self):
        """Calculate additional fight metrics for better target creation"""
        # First, log available columns
        logging.info("Available columns in events_df:")
        for col in self.events_df.columns:
            logging.info(f"- {col}")
        
        # Calculate striking pace (this should work with existing columns)
        self.events_df['combined_striking_pace'] = (
            self.events_df['r_SLpM_total'] + self.events_df['b_SLpM_total']
        )
        
        # Calculate experience level
        self.events_df['combined_experience'] = (
            self.events_df['r_wins_total'] + self.events_df['r_losses_total'] +
            self.events_df['b_wins_total'] + self.events_df['b_losses_total']
        )
        
        # Calculate win percentages instead of finish rates
        self.events_df['r_win_percentage'] = (
            self.events_df['r_wins_total'] / 
            (self.events_df['r_wins_total'] + self.events_df['r_losses_total'])
        ).fillna(0)
        
        self.events_df['b_win_percentage'] = (
            self.events_df['b_wins_total'] / 
            (self.events_df['b_wins_total'] + self.events_df['b_losses_total'])
        ).fillna(0)
        
        return self.events_df

    def _create_watchability_target(self):
        """Create binary target variable for fight watchability with improved criteria"""
        logging.info("Creating watchability target variable...")
        start_time = time.time()
        
        # Calculate additional metrics
        self._calculate_fight_metrics()
        
        # Create base excitement score from best fights
        best_fights_fighters = set()
        for _, row in tqdm(self.best_fights_df.iterrows(), total=len(self.best_fights_df)):
            best_fights_fighters.add((row['r_fighter'], row['b_fighter']))
            best_fights_fighters.add((row['b_fighter'], row['r_fighter']))
        
        self.events_df['is_best_fight_base'] = self.events_df.apply(
            lambda row: (row['r_fighter'], row['b_fighter']) in best_fights_fighters, 
            axis=1
        )
        
        # Create multi-criteria excitement definition
        self.events_df['is_exciting'] = (
            (self.events_df['is_best_fight_base']) |  # Original best fights
            (self.events_df['r_win_percentage'] >= 0.8) |  # High win rate (red corner)
            (self.events_df['b_win_percentage'] >= 0.8) |  # High win rate (blue corner)
            (self.events_df['combined_striking_pace'] >= 10) |  # High action fights
            (self.events_df['is_title_bout'] == True) |  # Title fights
            (self.events_df['combined_experience'] >= 30) |  # Veteran fighters
            (
                (self.events_df['r_wins_total'] >= 10) & 
                (self.events_df['b_wins_total'] >= 10)  # Both experienced fighters
            )
        )
        
        # Set final target
        self.events_df['is_best_fight'] = self.events_df['is_exciting']
        
        # Log distribution statistics
        exciting_fights = self.events_df['is_best_fight'].sum()
        total_fights = len(self.events_df)
        
        logging.info(f"\nExcitement criteria breakdown:")
        logging.info(f"Original best fights: {self.events_df['is_best_fight_base'].sum()}")
        logging.info(f"High red corner win rate: {(self.events_df['r_win_percentage'] >= 0.8).sum()}")
        logging.info(f"High blue corner win rate: {(self.events_df['b_win_percentage'] >= 0.8).sum()}")
        logging.info(f"High striking pace: {(self.events_df['combined_striking_pace'] >= 10).sum()}")
        logging.info(f"Title fights: {self.events_df['is_title_bout'].sum()}")
        logging.info(f"Veteran matchups: {(self.events_df['combined_experience'] >= 30).sum()}")
        logging.info(f"\nTotal exciting fights: {exciting_fights} ({exciting_fights/total_fights:.1%})")
        
        return self.events_df

    def select_features(self):
        """Select and engineer relevant features"""
        logging.info("Selecting and preparing features...")
        start_time = time.time()
        
        # Log available columns first
        logging.info("Available columns:")
        for col in self.events_df.columns:
            logging.info(f"- {col}")
        
        # Core fighter stats (check which ones exist)
        base_fighter_features = [
            'r_wins_total', 'r_losses_total', 'r_age', 'r_height', 'r_weight',
            'r_SLpM_total', 'r_SApM_total', 'r_sig_str_acc_total',
            'r_td_acc_total', 'r_str_def_total', 'r_td_def_total',
            'r_win_percentage',  # New feature
            'b_wins_total', 'b_losses_total', 'b_age', 'b_height', 'b_weight',
            'b_SLpM_total', 'b_SApM_total', 'b_sig_str_acc_total',
            'b_td_acc_total', 'b_str_def_total', 'b_td_def_total',
            'b_win_percentage'  # New feature
        ]
        
        # Filter to only include columns that exist
        fighter_features = [col for col in base_fighter_features if col in self.events_df.columns]
        logging.info(f"Using {len(fighter_features)} fighter features")
        
        # Base difference features
        base_diff_features = [
            'wins_total_diff', 'losses_total_diff', 'age_diff',
            'height_diff', 'weight_diff', 'SLpM_total_diff',
            'SApM_total_diff', 'sig_str_acc_total_diff',
            'td_acc_total_diff', 'str_def_total_diff', 'td_def_total_diff'
        ]
        
        # Filter difference features
        diff_features = [col for col in base_diff_features if col in self.events_df.columns]
        logging.info(f"Using {len(diff_features)} difference features")
        
        # Base fight features
        base_fight_features = [
            'is_title_bout', 'total_rounds',
            'combined_striking_pace',  # New feature
            'combined_experience'      # New feature
        ]
        
        # Filter fight features
        fight_features = [col for col in base_fight_features if col in self.events_df.columns]
        logging.info(f"Using {len(fight_features)} fight features")
        
        # Create initial feature DataFrame
        features_df = self.events_df[fighter_features + diff_features + fight_features].copy()
        
        # Check for division information in the dataset
        division_cols = [col for col in self.events_df.columns if 'division' in col.lower()]
        weight_class_cols = [col for col in self.events_df.columns if 'weight_class' in col.lower()]
        
        if division_cols:
            logging.info(f"Found division columns: {division_cols}")
            # One-hot encode division if it exists
            division_dummies = pd.get_dummies(self.events_df[division_cols[0]], prefix='division')
            features_df = pd.concat([features_df, division_dummies], axis=1)
        elif weight_class_cols:
            logging.info(f"Found weight class columns: {weight_class_cols}")
            # One-hot encode weight class if it exists
            weight_dummies = pd.get_dummies(self.events_df[weight_class_cols[0]], prefix='weight_class')
            features_df = pd.concat([features_df, weight_dummies], axis=1)
        
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        logging.info(f"Final feature matrix shape: {features_df.shape}")
        logging.info("Final features included:")
        for col in features_df.columns:
            logging.info(f"- {col}")
        
        return features_df, self.events_df['is_best_fight']

    # [Rest of the class methods remain the same]    
    def balance_dataset(self, X, y):
        """
        Balance dataset using SMOTE (only for training data)
        """
        logging.info("Balancing the dataset using SMOTE...")
        start_time = time.time()
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        logging.info(f"Balanced dataset created in {time.time() - start_time:.2f} seconds")
        logging.info(f"Positive class distribution after balancing: {y_balanced.mean():.2%}")
        return X_balanced, y_balanced
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess data and split into train/test sets
        """
        logging.info("Starting data preprocessing...")
        preprocessing_start = time.time()
        X, y = self.select_features()
        logging.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Apply SMOTE only to the training set
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train)
        
        logging.info(f"Train set shape after balancing: {X_train_balanced.shape}")
        
        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Save processed data and scaler
        processed_dir = os.path.join(self.base_dir, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save data
        pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
            os.path.join(processed_dir, 'X_train.csv'), index=False)
        pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
            os.path.join(processed_dir, 'X_test.csv'), index=False)
        pd.Series(y_train_balanced).to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
        pd.Series(y_test).to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
        
        # Save scaler
        models_dir = os.path.join(self.base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        
        logging.info(f"Data preprocessing completed in {time.time() - preprocessing_start:.2f} seconds")
        return X_train_scaled, X_test_scaled, y_train_balanced, y_test

# Example usage
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        preprocessor = UFCFightPreprocessor(base_dir)
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
        logging.info("Preprocessing script completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        import traceback
        logging.error(traceback.format_exc())

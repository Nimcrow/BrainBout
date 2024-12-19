import pandas as pd
import numpy as np
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler

class UFCMatchmaker:
    def __init__(self, roster_path, model_dir):
        """
        Initialize the UFC Matchmaker
        
        Parameters:
        - roster_path: Path to the UFC roster CSV file
        - model_dir: Directory containing the model and feature names
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load data and model
        try:
            self.roster_df = pd.read_csv(roster_path)
            self.logger.info(f"Successfully loaded {len(self.roster_df)} fighters from roster")
            
            # Load model, scaler and feature names
            model_path = os.path.join(model_dir, 'random_forest_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            feature_path = os.path.join(model_dir, 'feature_names.csv')
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = pd.read_csv(feature_path).columns.tolist()
            
            self.logger.info(f"Loaded model with {len(self.feature_names)} features")
            
            # Clean and prepare roster data
            self._prepare_roster_data()
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise
            
    def _prepare_roster_data(self):
        """Prepare and clean the roster data"""
        # Convert string percentages to floats
        percentage_columns = ['striking_accuracy', 'takedown_accuracy']
        for col in percentage_columns:
            self.roster_df[col] = pd.to_numeric(self.roster_df[col].str.rstrip('%'), errors='coerce') / 100
            
        # Extract win numbers from record
        self.roster_df['wins_total'] = self.roster_df['record'].str.extract(r'(\d+)-').astype(float)
        self.roster_df['losses_total'] = self.roster_df['record'].str.extract(r'-(\d+)-').astype(float)
        
        # Convert height to inches (if not already)
        self.roster_df['height'] = pd.to_numeric(self.roster_df['height'], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['age', 'weight', 'sig_str_landed_per_min', 'sig_str_absorbed_per_min',
                         'takedown_avg_per_15_min', 'submission_avg_per_15_min',
                         'height', 'reach', 'leg_reach', 'sig_str_defense', 'takedown_defense']
        
        # Convert each numeric column
        for col in numeric_columns:
            if col in self.roster_df.columns:
                self.roster_df[col] = pd.to_numeric(self.roster_df[col], errors='coerce')
        
        # Fill missing values with median
        self.roster_df[numeric_columns] = self.roster_df[numeric_columns].fillna(
            self.roster_df[numeric_columns].median()
        )
    
    def _prepare_fight_features(self, fighter1, fighter2):
        """
        Prepare features for fight prediction with proper scaling
        """
        # Create base features dictionary
        features = {
            # Fighter 1 (Red Corner) features
            'r_wins_total': float(fighter1['wins_total'] if pd.notnull(fighter1['wins_total']) else 0),
            'r_losses_total': float(fighter1['losses_total'] if pd.notnull(fighter1['losses_total']) else 0),
            'r_age': float(fighter1['age'] if pd.notnull(fighter1['age']) else 0),
            'r_height': float(fighter1['height'] if pd.notnull(fighter1['height']) else 0),
            'r_weight': float(fighter1['weight'] if pd.notnull(fighter1['weight']) else 0),
            'r_SLpM_total': float(fighter1['sig_str_landed_per_min'] if pd.notnull(fighter1['sig_str_landed_per_min']) else 0),
            'r_SApM_total': float(fighter1['sig_str_absorbed_per_min'] if pd.notnull(fighter1['sig_str_absorbed_per_min']) else 0),
            'r_sig_str_acc_total': float(fighter1['striking_accuracy'] if pd.notnull(fighter1['striking_accuracy']) else 0),
            'r_td_acc_total': float(fighter1['takedown_accuracy'] if pd.notnull(fighter1['takedown_accuracy']) else 0),
            'r_str_def_total': float(fighter1['sig_str_defense'] if pd.notnull(fighter1['sig_str_defense']) else 0),
            'r_td_def_total': float(fighter1['takedown_defense'] if pd.notnull(fighter1['takedown_defense']) else 0),
        
            # Fighter 2 (Blue Corner) features
            'b_wins_total': float(fighter2['wins_total'] if pd.notnull(fighter2['wins_total']) else 0),
            'b_losses_total': float(fighter2['losses_total'] if pd.notnull(fighter2['losses_total']) else 0),
            'b_age': float(fighter2['age'] if pd.notnull(fighter2['age']) else 0),
            'b_height': float(fighter2['height'] if pd.notnull(fighter2['height']) else 0),
            'b_weight': float(fighter2['weight'] if pd.notnull(fighter2['weight']) else 0),
            'b_SLpM_total': float(fighter2['sig_str_landed_per_min'] if pd.notnull(fighter2['sig_str_landed_per_min']) else 0),
            'b_SApM_total': float(fighter2['sig_str_absorbed_per_min'] if pd.notnull(fighter2['sig_str_absorbed_per_min']) else 0),
            'b_sig_str_acc_total': float(fighter2['striking_accuracy'] if pd.notnull(fighter2['striking_accuracy']) else 0),
            'b_td_acc_total': float(fighter2['takedown_accuracy'] if pd.notnull(fighter2['takedown_accuracy']) else 0),
            'b_str_def_total': float(fighter2['sig_str_defense'] if pd.notnull(fighter2['sig_str_defense']) else 0),
            'b_td_def_total': float(fighter2['takedown_defense'] if pd.notnull(fighter2['takedown_defense']) else 0),
        }
    
        # Calculate differences
        features.update({
            'wins_total_diff': features['r_wins_total'] - features['b_wins_total'],
            'losses_total_diff': features['r_losses_total'] - features['b_losses_total'],
            'age_diff': features['r_age'] - features['b_age'],
            'height_diff': features['r_height'] - features['b_height'],
            'weight_diff': features['r_weight'] - features['b_weight'],
            'SLpM_total_diff': features['r_SLpM_total'] - features['b_SLpM_total'],
            'SApM_total_diff': features['r_SApM_total'] - features['b_SApM_total'],
            'sig_str_acc_total_diff': features['r_sig_str_acc_total'] - features['b_sig_str_acc_total'],
            'td_acc_total_diff': features['r_td_acc_total'] - features['b_td_acc_total'],
            'str_def_total_diff': features['r_str_def_total'] - features['b_str_def_total'],
            'td_def_total_diff': features['r_td_def_total'] - features['b_td_def_total'],
        })

        # Create DataFrame with feature names
        features_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
    
        # Update the values we have
        for col in features:
            if col in features_df.columns:
                features_df[col] = features[col]

        # Set division-based features
        division = fighter1['division']
        weight_class = division.replace(' Division', '')  # Remove 'Division' suffix if present
    
        # Reset all weight class and gender columns to 0
        for col in features_df.columns:
            if col.startswith('weight_class_') or col.startswith('gender_'):
                features_df[col] = 0
    
        # Set appropriate weight class
        col_name = f'weight_class_{weight_class}'
        if col_name in features_df.columns:
            features_df[col_name] = 1
    
        # Set gender
        if "Women's" in division:
            if 'gender_Female' in features_df.columns:
                features_df['gender_Female'] = 1
            if 'gender_Male' in features_df.columns:
                features_df['gender_Male'] = 0
        else:
            if 'gender_Female' in features_df.columns:
                features_df['gender_Female'] = 0
            if 'gender_Male' in features_df.columns:
                features_df['gender_Male'] = 1
    
        # Set basic fight features
        if 'is_title_bout' in features_df.columns:
            features_df['is_title_bout'] = 0  # Default to non-title bout
        if 'total_rounds' in features_df.columns:
            features_df['total_rounds'] = 3   # Default to 3 rounds

        # Verify all features are present
        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        # Keep only needed features in correct order
        features_df = features_df[self.feature_names]

        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Scale the features using the same scaler used during training
        scaled_features = self.scaler.transform(features_df)
        
        return scaled_features
    
    def _find_fighter(self, name):
        """
        Find a fighter using fuzzy name matching
        """
        # Convert input name to lowercase for comparison
        name_lower = name.lower()
        
        # First try exact match (case-insensitive)
        matches = self.roster_df[self.roster_df['name'].str.lower() == name_lower]
        
        if not matches.empty:
            return matches.iloc[0]
            
        # If no exact match, try contains
        matches = self.roster_df[self.roster_df['name'].str.lower().str.contains(name_lower, na=False)]
        
        if matches.empty:
            return None
        elif len(matches) == 1:
            return matches.iloc[0]
        else:
            # Multiple matches found
            return matches

    def analyze_matchup(self, fighter1_name, fighter2_name):
        """
        Analyze a potential matchup between two fighters
        
        Parameters:
        - fighter1_name: Name of first fighter
        - fighter2_name: Name of second fighter
        
        Returns:
        - Dictionary containing prediction and analysis
        """
        try:
            # Get fighter data with fuzzy matching
            fighter1_result = self._find_fighter(fighter1_name)
            fighter2_result = self._find_fighter(fighter2_name)
            
            # Handle multiple matches
            if fighter1_result is None:
                return {'error': f"Fighter not found: {fighter1_name}"}
            elif isinstance(fighter1_result, pd.DataFrame):
                matches = fighter1_result['name'].tolist()
                return {'error': f"Multiple matches found for '{fighter1_name}'. Did you mean one of these?\n" + "\n".join(f"  • {name}" for name in matches)}
            
            if fighter2_result is None:
                return {'error': f"Fighter not found: {fighter2_name}"}
            elif isinstance(fighter2_result, pd.DataFrame):
                matches = fighter2_result['name'].tolist()
                return {'error': f"Multiple matches found for '{fighter2_name}'. Did you mean one of these?\n" + "\n".join(f"  • {name}" for name in matches)}
            
            fighter1 = fighter1_result
            fighter2 = fighter2_result
            
            # Check if fighters are in same division
            if fighter1['division'] != fighter2['division']:
                return {
                    'error': f"Fighters are in different divisions: {fighter1['division']} vs {fighter2['division']}"
                }
            
            # Prepare features
            features = self._prepare_fight_features(fighter1, fighter2)
            
            # Debug: Print feature statistics before prediction
            self.logger.info("\nFeature statistics before prediction:")
            self.logger.info(f"Shape: {features.shape}")
            self.logger.info(f"Mean: {np.mean(features)}")
            self.logger.info(f"Std: {np.std(features)}")
            self.logger.info(f"Min: {np.min(features)}")
            self.logger.info(f"Max: {np.max(features)}")
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            prediction_prob = self.model.predict_proba(features)[0][1]
            
            # Debug: Print prediction details
            self.logger.info("\nPrediction details:")
            self.logger.info(f"Raw prediction: {prediction}")
            self.logger.info(f"Probability: {prediction_prob}")
            
            # Generate analysis
            analysis = self._generate_analysis(fighter1, fighter2, prediction_prob)
            
            return {
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'division': fighter1['division'],
                'is_exciting': bool(prediction),
                'excitement_probability': prediction_prob,
                'analysis': analysis
            }
            
        except IndexError:
            return {'error': f"One or both fighters not found: {fighter1_name}, {fighter2_name}"}
        except Exception as e:
            return {'error': f"Error analyzing matchup: {str(e)}"}
    
    def _generate_analysis(self, fighter1, fighter2, excitement_prob):
        """Generate detailed analysis of the matchup with more nuanced metrics"""
        analysis = []
        
        # Striking analysis with more nuanced thresholds
        striking_diff = abs(float(fighter1['sig_str_landed_per_min']) - float(fighter2['sig_str_landed_per_min']))
        if striking_diff > 3:
            analysis.append(f"Very high striking differential ({striking_diff:.1f} strikes/min) suggests potential for action")
        elif striking_diff > 1.5:
            analysis.append(f"Moderate striking differential ({striking_diff:.1f} strikes/min)")
            
        # Combined striking volume
        total_striking = float(fighter1['sig_str_landed_per_min']) + float(fighter2['sig_str_landed_per_min'])
        if total_striking > 8:
            analysis.append("High combined striking volume suggests a stand-up war")
        
        # Style matchup analysis
        try:
            f1_td = float(fighter1['takedown_avg_per_15_min'])
            f2_td = float(fighter2['takedown_avg_per_15_min'])
            f1_td_def = float(fighter1['takedown_defense'])
            f2_td_def = float(fighter2['takedown_defense'])
            
            # Grappling heavy matchup
            if f1_td > 2 and f2_td > 2:
                analysis.append("Both fighters actively seek takedowns - potential for dynamic grappling exchanges")
            # Striker vs Grappler
            elif abs(f1_td - f2_td) > 2:
                if f1_td > f2_td:
                    if f2_td_def > 80:
                        analysis.append("Classic striker vs grappler matchup with strong takedown defense")
                    else:
                        analysis.append("Potential for ground control from the wrestler")
                else:
                    if f1_td_def > 80:
                        analysis.append("Classic striker vs grappler matchup with strong takedown defense")
                    else:
                        analysis.append("Potential for ground control from the wrestler")
        except (ValueError, TypeError):
            pass
            
        # Finishing analysis without using wins_by_Decision
        try:
            f1_finish_rate = float(fighter1['finish_rate']) if 'finish_rate' in fighter1 else None
            f2_finish_rate = float(fighter2['finish_rate']) if 'finish_rate' in fighter2 else None
            
            if f1_finish_rate is not None and f2_finish_rate is not None:
                avg_finish_rate = (f1_finish_rate + f2_finish_rate) / 2
                if avg_finish_rate > 0.7:
                    analysis.append("Both fighters have strong finishing ability")
                elif avg_finish_rate > 0.5:
                    analysis.append("Moderate finishing rate between both fighters")
        except (ValueError, TypeError):
            pass
            
        # Experience and momentum
        try:
            f1_recent = float(fighter1['wins_total']) / (float(fighter1['wins_total']) + float(fighter1['losses_total']))
            f2_recent = float(fighter2['wins_total']) / (float(fighter2['wins_total']) + float(fighter2['losses_total']))
            
            if abs(f1_recent - f2_recent) > 0.3:
                analysis.append("Significant momentum difference between fighters")
            elif abs(f1_recent - f2_recent) < 0.1:
                analysis.append("Both fighters showing similar recent success")
        except (ValueError, TypeError, ZeroDivisionError):
            pass
            
        if not analysis:
            analysis.append("Limited statistical comparison available")
            
        return analysis

def main():
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roster_path = os.path.join(base_dir, 'data', 'raw', 'ufc_fighters_roster_stats_and_records.csv')
    model_dir = os.path.join(base_dir, 'models')

    print("\nStarting UFC Fight Excitement Predictor...")
    try:
        # Initialize matchmaker
        matchmaker = UFCMatchmaker(roster_path, model_dir)
        
        while True:
            print("\nEnter fighter names (or 'quit' to exit, 'list' to see all fighters)")
            fighter1_name = input("Fighter 1: ").strip()
            
            if fighter1_name.lower() == 'quit':
                break
                
            if fighter1_name.lower() == 'list':
                active_fighters = matchmaker.roster_df[
                    (matchmaker.roster_df['status'] == 'Active') &
                    (matchmaker.roster_df['division'].notna())
                ]
                print("\nActive fighters by division:")
                for division in sorted(active_fighters['division'].unique()):
                    fighters = active_fighters[active_fighters['division'] == division]['name'].tolist()
                    print(f"\n{division}:")
                    for fighter in sorted(fighters):
                        print(f"  • {fighter}")
                continue
                
            fighter2_name = input("Fighter 2: ").strip()
            
            if fighter2_name.lower() == 'quit':
                break
                
            if fighter2_name.lower() == 'list':
                active_fighters = matchmaker.roster_df[
                    (matchmaker.roster_df['status'] == 'Active') &
                    (matchmaker.roster_df['division'].notna())
                ]
                print("\nActive fighters by division:")
                for division in sorted(active_fighters['division'].unique()):
                    fighters = active_fighters[active_fighters['division'] == division]['name'].tolist()
                    print(f"\n{division}:")
                    for fighter in sorted(fighters):
                        print(f"  • {fighter}")
                continue
            
            result = matchmaker.analyze_matchup(fighter1_name, fighter2_name)
            
            if 'error' in result:
                print(f"\n{result['error']}")
                continue
            
            print(f"\nMatchup Analysis: {result['fighter1']} vs {result['fighter2']}")
            print(f"Division: {result['division']}")
            print("-" * 50)
            print(f"Prediction: {'EXCITING! 🔥' if result['is_exciting'] else 'Less Exciting'}")
            print(f"Confidence: {result['excitement_probability']:.2%}")
            print("\nAnalysis:")
            for point in result['analysis']:
                print(f"• {point}")
            
    except Exception as e:
        logging.error(f"Error running matchmaker: {str(e)}")
        raise

if __name__ == "__main__":
    main()
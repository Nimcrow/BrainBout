import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

class UFCFightPredictor:
    def __init__(self, X_train, X_test, y_train, y_test, base_dir=None, log_level=logging.INFO):
        """
        Initialize predictor with training and test data
        
        Parameters:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training labels
        - y_test: Test labels
        - base_dir: Base directory for saving outputs
        - log_level: Logging level (default: INFO)
        """
        # Setup logging
        logging.basicConfig(
            level=log_level, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set base directory
        self.base_dir = base_dir if base_dir is not None else os.getcwd()
        
        # Log initial data dimensions
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Test data shape: {X_test.shape}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Store column names
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        
        # Log class distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        self.logger.info("Training data class distribution:")
        for cls, count in zip(unique_train, counts_train):
            self.logger.info(f"Class {cls}: {count} samples")
        
        self.logger.info("Test data class distribution:")
        for cls, count in zip(unique_test, counts_test):
            self.logger.info(f"Class {cls}: {count} samples")
        
        # # Initialize models
        # self.models = {
        #     'Logistic Regression': {
        #         'model': LogisticRegression(random_state=42),
        #         'params': {
        #             'C': [0.001, 0.01, 0.1, 1, 10, 100],
        #             'penalty': ['l1', 'l2'],
        #             'solver': ['liblinear']
        #         }
        #     },
        #     'Random Forest': {
        #         'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        #         'params': {
        #             'n_estimators': [100, 200],  # More trees
        #             'max_depth': [None, 10, 20, 30],
        #             'min_samples_split': [2, 5],
        #             'min_samples_leaf': [1, 2],  # Add this parameter
        #             'class_weight': ['balanced', 'balanced_subsample'],  # Handle class imbalance
        #             'max_features': ['auto', 'sqrt', 'log2']  # Add feature selection methods
        #         }
        #     },
        #     'SVM': {
        #         'model': SVC(probability=True, random_state=42),
        #         'params': {
        #             'C': [0.1, 1, 10, 100],
        #             'kernel': ['linear'],
        #             'gamma': ['scale', 'auto']
        #         }
        #     }
        # }

        # Initialize models with better parameters and calibration
        # self.models = {
        #     'Logistic Regression': {
        #         'model': LogisticRegression(
        #             random_state=42,
        #             class_weight='balanced',
        #             max_iter=1000
        #         ),
        #         'params': {
        #             'C': [0.01, 0.1, 1, 10],
        #             'penalty': ['l2'],
        #             'solver': ['lbfgs']
        #         }
        #     },
        #     'Random Forest': {
        #         'model':RandomForestClassifier(
        #                 random_state=42,
        #                 n_jobs=-1,
        #                 class_weight='balanced',
        #                 n_estimators=200,
        #                 min_samples_leaf=5  # Prevent overfitting
        #             ),
        #             'params': {
        #                 'max_depth': [10, 20, None],
        #                 'min_samples_split': [5, 10],
        #                 'max_features': ['sqrt', 'log2']
        #             }
        #     },
        #     'SVM': {
        #         'model': SVC(
        #             random_state=42, 
        #             class_weight='balanced'
        #         ),
        #         'params': {
        #             'C': [1, 10],
        #             'kernel': ['rbf'],
        #             'gamma': ['scale']
        #         }
        #     }
        # }
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'newton-cg']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=500,  # Increased number of trees
                    min_samples_leaf=3,
                    min_samples_split=6,
                    max_features='sqrt',
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True,  # Enable bootstrapping
                    oob_score=True   # Use out-of-bag score
                ),
                'params': {
                    'max_depth': [20, 30, None],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_leaf': [2, 3, 4],
                    'min_samples_split': [5, 6, 8]
                }
            },
            'SVM': {
                'model': SVC(
                    random_state=42, 
                    class_weight='balanced',
                    probability=True
                ),
                'params': {
                    'C': [1.0, 10.0, 100.0],
                    'kernel': ['rbf'],
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3]
                }
            }
        }
        
        # Results storage
        self.results = {}
    
    def evaluate_model(self, y_true, y_pred, y_prob):
        """
        Compute and return various evaluation metrics
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted labels
        - y_prob: Predicted probabilities
        
        Returns:
        - Dictionary of evaluation metrics
        """
        try:
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1 Score': f1_score(y_true, y_pred, zero_division=0),
                'ROC AUC': roc_auc_score(y_true, y_prob)
            }
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            return {}
    
    def grid_search_and_train(self):
        """
        Perform grid search for hyperparameter tuning and train models
        """
        for name, model_info in self.models.items():
            self.logger.info(f"Starting Grid Search for {name}")

            # Convert to numpy arrays for training if needed
            X_train = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            X_test = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
            
            # Perform Grid Search
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X_train, self.y_train)

            # Best model
            best_model = grid_search.best_estimator_
            
            # Log OOB score and feature importance before calibration for Random Forest
            if name == 'Random Forest':
                self.logger.info(f"Random Forest Out-of-bag score: {best_model.oob_score_:.3f}")
                
                if hasattr(best_model, 'feature_importances_'):
                    importances = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    self.logger.info("\nTop 10 most important features:")
                    for _, row in importances.head(10).iterrows():
                        self.logger.info(f"{row['feature']}: {row['importance']:.4f}")

            # Apply calibration after finding the best model
            if name in ['Random Forest', 'SVM']:
                best_model = CalibratedClassifierCV(
                    best_model, 
                    cv=5, 
                    method='sigmoid'
                ).fit(X_train, self.y_train)
        
            self.logger.info(f"Best Parameters for {name}: {grid_search.best_params_}")
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Store results
            self.results[name] = {
                'Best Parameters': grid_search.best_params_,
                'Metrics': self.evaluate_model(self.y_test, y_pred, y_prob),
                'Confusion Matrix': confusion_matrix(self.y_test, y_pred),
                'Model': best_model
            }
            
            # Log model metrics
            self.logger.info(f"Model Performance Metrics for {name}:")
            for metric, value in self.results[name]['Metrics'].items():
                self.logger.info(f"{metric}: {value}")

            # Save the Random Forest model and feature names
            if name == "Random Forest":
                model_dir = os.path.join(self.base_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
            
                # Save model
                model_path = os.path.join(model_dir, 'random_forest_model.pkl')
                joblib.dump(best_model, model_path)
                self.logger.info(f"Random Forest model saved to {model_path}")
            
                # Save feature names if available
                if self.feature_names:
                    feature_names_df = pd.DataFrame(columns=self.feature_names)
                    feature_path = os.path.join(model_dir, 'feature_names.csv')
                    feature_names_df.to_csv(feature_path, index=False)
                    self.logger.info(f"Feature names saved to {feature_path}")
                else:
                    self.logger.warning("No feature names available to save")
    
    def visualize_results(self, base_dir=None):
        """
        Create visualizations of model performance
        
        Parameters:
        - base_dir: Base directory of the project
        """
        if base_dir is None:
            base_dir = self.base_dir
        
        # Create figures directory
        figures_dir = os.path.join(base_dir, 'notebooks', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        self.logger.info(f"Creating visualizations in {figures_dir}")
        
        # Metrics comparison
        metrics_df = pd.DataFrame([
            {name: metrics['Metrics'][metric] 
             for name, metrics in self.results.items()}
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        ], index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
        
        # Plot metrics
        plt.figure(figsize=(12, 6))
        metrics_df.T.plot(kind='bar', rot=45)
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'model_performance.png'))
        plt.close()
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        for name, model_results in self.results.items():
            y_prob = model_results['Model'].predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {model_results["Metrics"]["ROC AUC"]:.2f})')
        
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Models')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'roc_curves.png'))
        plt.close()
    
    def save_results(self, base_dir=None):
        """
        Save model results and performance metrics
        
        Parameters:
        - base_dir: Base directory of the project
        """
        if base_dir is None:
            base_dir = self.base_dir
        
        # Create outputs directory
        outputs_dir = os.path.join(base_dir, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(outputs_dir, 'model_results.txt')
        self.logger.info(f"Saving model results to {results_path}")
        
        with open(results_path, 'w') as f:
            for name, results in self.results.items():
                f.write(f"Model: {name}\n")
                f.write("Best Parameters:\n")
                f.write(str(results['Best Parameters']) + "\n\n")
                f.write("Performance Metrics:\n")
                for metric, value in results['Metrics'].items():
                    f.write(f"{metric}: {value}\n")
                f.write("\n" + "="*50 + "\n\n")
    
    def run(self, base_dir=None):
        """
        Run full model training and evaluation pipeline
        """
        self.logger.info("Starting model training and evaluation pipeline")
        
        # Perform grid search and training
        self.grid_search_and_train()
        
        # Visualize results
        self.visualize_results(base_dir)
        
        # Save results
        self.save_results(base_dir)
        
        # Find and log best model
        best_model = max(self.results.items(), key=lambda x: x[1]['Metrics']['F1 Score'])
        self.logger.info(f"Best Model: {best_model[0]}")
        self.logger.info(f"Best Model Performance: {best_model[1]['Metrics']}")
        
        return self.results

# Example usage
if __name__ == '__main__':
    # Automatically detect base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load preprocessed data
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    logging.info(f"Loading data from {processed_dir}")
    
    # Load data preserving column names
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
    
    # Initialize and run predictor
    predictor = UFCFightPredictor(X_train, X_test, y_train, y_test, base_dir=base_dir)
    results = predictor.run()
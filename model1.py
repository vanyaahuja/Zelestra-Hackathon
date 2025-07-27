# Solar Panel Efficiency Prediction Model - FULLY FIXED VERSION
# Handles all edge cases and data inconsistencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not available. Install with: pip install catboost")
    CATBOOST_AVAILABLE = False


# STEP 1: LOADING DATASET


def load_and_explore_data(train_path, test_path):
    """Load and perform initial data exploration"""

    try:
        # Load datasets 
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print("=== DATA OVERVIEW ===")
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"\nTrain columns: {list(train_df.columns)}")
        print(f"Test columns: {list(test_df.columns)}")

        # Check data types
        print(f"\nTrain data types:")
        print(train_df.dtypes)

        # Check for any obvious issues
        print(f"\nChecking for non-numeric data in numeric columns...")
        numeric_cols = ['temperature', 'irradiance', 'humidity', 'voltage', 'current', 'module_temperature']
        for col in numeric_cols:
            if col in train_df.columns:
                non_numeric = train_df[col].apply(lambda x: not isinstance(x, (int, float, type(None))))
                if non_numeric.any():
                    print(f"Found non-numeric data in {col}: {train_df[col][non_numeric].unique()}")

        # Checking for data type issues in the datasets
        print("\n=== MISSING VALUES ===")
        missing_train = train_df.isnull().sum()
        missing_test = test_df.isnull().sum()

        print("Training set missing values:")
        print(missing_train[missing_train > 0])

        print("\nTest set missing values:")
        print(missing_test[missing_test > 0])

        return train_df, test_df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# STEP 2: FEATURE ENGINEERING


def safe_numeric_operation(series1, series2, operation='multiply', fill_value=0):
    """Safely perform numeric operations on potentially mixed-type series"""

    try:
        # Convert to numeric, errors='coerce' will turn non-numeric to NaN
        s1 = pd.to_numeric(series1, errors='coerce').fillna(fill_value)
        s2 = pd.to_numeric(series2, errors='coerce').fillna(fill_value)

        if operation == 'multiply':
            return s1 * s2
        elif operation == 'divide':
            # Add small epsilon to avoid division by zero
            return s1 / (s2 + 1e-8)
        elif operation == 'subtract':
            return s1 - s2
        elif operation == 'add':
            return s1 + s2
        else:
            return s1

    except Exception as e:
        print(f"Error in safe_numeric_operation: {e}")
        return pd.Series([fill_value] * len(series1), index=series1.index)

def analyze_feature_importance(X, y):
    """Analyze feature importance using Random Forest"""
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    # feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features
    top_features = importance[importance['importance'] > importance['importance'].mean()]['feature'].tolist()
    
    return top_features, importance

def create_feature_interactions(df, important_features):
    """Create interactions between important features"""
    interactions = {}
    
    # CreatING polynomial features for important numeric features
    numeric_features = df[important_features].select_dtypes(include=[np.number]).columns
    for i, feat1 in enumerate(numeric_features):
        for feat2 in numeric_features[i+1:]:
            # interaction
            interaction_name = f"{feat1}_{feat2}_interaction"
            interactions[interaction_name] = df[feat1] * df[feat2]
            
            # ratio if denominator is not zero
            ratio_name = f"{feat1}_{feat2}_ratio"
            interactions[ratio_name] = df[feat1] / (df[feat2] + 1e-8)
            
            # difference
            diff_name = f"{feat1}_{feat2}_diff"
            interactions[diff_name] = df[feat1] - df[feat2]
    
    return interactions

def create_solar_features(df):
    """Create solar-specific engineered features with correlation analysis and error code features"""

    df_eng = df.copy()

    print(f"Starting feature engineering on {df_eng.shape[0]} rows...")

    # Required columns for basic features
    required_cols = ['voltage', 'current', 'irradiance', 'module_temperature', 'temperature']
    available_cols = [col for col in required_cols if col in df_eng.columns]
    missing_cols = [col for col in required_cols if col not in df_eng.columns]

    if missing_cols:
        print(f" Missing columns: {missing_cols}")

    print(f" Available columns: {available_cols}")

    try:
        # 1. Power output (most important solar feature)
        if 'voltage' in df_eng.columns and 'current' in df_eng.columns:
            df_eng['power_output'] = safe_numeric_operation(df_eng['voltage'], df_eng['current'], 'multiply')
            print("Created power_output")
        else:
            df_eng['power_output'] = 0
            print("Created dummy power_output (missing voltage/current)")

        # 2. Error code analysis 
        if 'error_code' in df_eng.columns:
            # Creating error code categories
            df_eng['error_severity'] = df_eng['error_code'].apply(lambda x: 
                3 if pd.notna(x) and str(x).startswith('E') else  # Critical errors
                2 if pd.notna(x) and str(x).startswith('W') else  # Warnings
                1 if pd.notna(x) and str(x).startswith('I') else  # Information
                0)  # No error

            # Create error frequency 
            error_counts = df_eng.groupby('error_code').size()
            df_eng['error_frequency'] = df_eng['error_code'].map(error_counts).fillna(0)

            # Create error type 
            df_eng['has_voltage_error'] = df_eng['error_code'].str.contains('VOLTAGE', case=False, na=False).astype(int)
            df_eng['has_current_error'] = df_eng['error_code'].str.contains('CURRENT', case=False, na=False).astype(int)
            df_eng['has_temperature_error'] = df_eng['error_code'].str.contains('TEMP', case=False, na=False).astype(int)
            print("Created error code features")

        # 3. Performance ratios with error consideration
        if 'irradiance' in df_eng.columns:
            df_eng['performance_ratio'] = safe_numeric_operation(df_eng['power_output'], df_eng['irradiance'], 'divide')
            # Adjust performance ratio based on error severity
            if 'error_severity' in df_eng.columns:
                df_eng['adjusted_performance'] = df_eng['performance_ratio'] * (1 - df_eng['error_severity'] * 0.1)
            print("Created performance ratios")
        else:
            df_eng['performance_ratio'] = 0
            df_eng['adjusted_performance'] = 0
            print("Created dummy performance ratios")

        # 4. Temperature effects with error consideration
        if 'module_temperature' in df_eng.columns and 'temperature' in df_eng.columns:
            df_eng['temp_difference'] = safe_numeric_operation(df_eng['module_temperature'], df_eng['temperature'], 'subtract')
            df_eng['temp_efficiency_factor'] = safe_numeric_operation(df_eng['module_temperature'], df_eng['temperature'], 'divide')
            
            # Create temperature ranges
            df_eng['temp_range'] = pd.cut(df_eng['module_temperature'], 
                                        bins=[-np.inf, 0, 25, 50, 75, np.inf],
                                        labels=['freezing', 'cold', 'optimal', 'warm', 'hot'])
            
            # Adjust temperature features based on error codes
            if 'has_temperature_error' in df_eng.columns:
                df_eng['temp_efficiency_factor'] = df_eng['temp_efficiency_factor'] * (1 - df_eng['has_temperature_error'] * 0.2)
            print("Created temperature features")
        else:
            df_eng['temp_difference'] = 0
            df_eng['temp_efficiency_factor'] = 1
            df_eng['temp_range'] = 'optimal'
            print("Created dummy temperature features")

        # 5. Voltage-Current relationship features
        if 'voltage' in df_eng.columns and 'current' in df_eng.columns:
            df_eng['voltage_current_ratio'] = safe_numeric_operation(df_eng['voltage'], df_eng['current'], 'divide')
            
            # Create voltage ranges
            df_eng['voltage_range'] = pd.cut(df_eng['voltage'],
                                           bins=[-np.inf, 0, 12, 24, 48, np.inf],
                                           labels=['negative', 'low', 'normal', 'high', 'very_high'])
            
            # Adjust based on voltage errors
            if 'has_voltage_error' in df_eng.columns:
                df_eng['voltage_current_ratio'] = df_eng['voltage_current_ratio'] * (1 - df_eng['has_voltage_error'] * 0.15)
            print("Created voltage-current features")
        else:
            df_eng['voltage_current_ratio'] = 1
            df_eng['voltage_range'] = 'normal'
            print("Created dummy voltage-current features")

        # 6. Environmental combinations with error consideration
        if 'cloud_coverage' in df_eng.columns and 'humidity' in df_eng.columns:
            df_eng['weather_impact'] = safe_numeric_operation(df_eng['cloud_coverage'], df_eng['humidity'], 'multiply')
            df_eng['weather_severity'] = pd.cut(df_eng['weather_impact'],
                                              bins=[-np.inf, 0, 25, 50, 75, 100],
                                              labels=['none', 'light', 'moderate', 'severe', 'extreme'])
            print("Created weather features")
        else:
            df_eng['weather_impact'] = 0
            df_eng['weather_severity'] = 'none'
            print("Created dummy weather features")

        # 7. Feature importance analysis and interactions
        if 'efficiency' in df_eng.columns:
            # Get numeric columns for analysis
            numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                # Analyze feature importance
                important_features, importance_df = analyze_feature_importance(
                    df_eng[numeric_cols], 
                    df_eng['efficiency']
                )
                
                print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
                print(importance_df.head(10))
                
                # Create interactions between important features
                interactions = create_feature_interactions(df_eng, important_features)
                
                # Add interactions to dataframe
                for name, values in interactions.items():
                    df_eng[name] = values
              
                print(f"Created {len(interactions)} feature interactions")
            else:
                print(" Not enough numeric columns for feature importance analysis")
        else:
            print("Target variable 'efficiency' not found for feature importance analysis")

        # 8. Error-based efficiency adjustments
        if 'error_severity' in df_eng.columns and 'performance_ratio' in df_eng.columns:
            df_eng['error_adjusted_efficiency'] = df_eng['performance_ratio'] * (1 - df_eng['error_severity'] * 0.1)
            print("Created error-adjusted efficiency")
        else:
            df_eng['error_adjusted_efficiency'] = df_eng['performance_ratio'] if 'performance_ratio' in df_eng.columns else 0
            print("Created dummy error-adjusted efficiency")

        # Clean up any infinite or extremely large values
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_eng[col] = df_eng[col].replace([np.inf, -np.inf], np.nan)
            # Cap extremely large values
            if df_eng[col].dtype in ['float64', 'int64']:
                q99 = df_eng[col].quantile(0.99)
                q01 = df_eng[col].quantile(0.01)
                if pd.notna(q99) and pd.notna(q01):
                    df_eng[col] = np.clip(df_eng[col], q01, q99)

        # Ensure categorical columns are strings
        cat_cols = ['temp_range', 'voltage_range', 'weather_severity']
        for col in cat_cols:
            if col in df_eng.columns:
                df_eng[col] = df_eng[col].astype(str)

        new_features = [col for col in df_eng.columns if col not in df.columns]
        print(f"Successfully created {len(new_features)} features: {new_features}")

        return df_eng

    except Exception as e:
        print(f"Critical error in feature engineering: {e}")
        # Return original dataframe if everything fails
        return df


# STEP 3: BULLETPROOF PREPROCESSING (work is going on guyz)

def preprocess_data(train_df, test_df, target_col='efficiency'):
    """Bulletproof data preprocessing"""

    try:
        print(f"Starting preprocessing...")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # Separate features and target
        if target_col in train_df.columns:
            X_train = train_df.drop([target_col], axis=1)
            y_train = train_df[target_col].copy()
            print(f"Found target variable: {target_col}")
        else:
            X_train = train_df.copy()
            y_train = None
            print(f"Target column '{target_col}' not found")

        X_test = test_df.copy()

        # Storing ID columns
        train_ids = X_train['id'].copy() if 'id' in X_train.columns else pd.Series(range(len(X_train)))
        test_ids = X_test['id'].copy() if 'id' in X_test.columns else pd.Series(range(len(X_test)))

        # Removing  ID from features
        for df_name, df in [('train', X_train), ('test', X_test)]:
            if 'id' in df.columns:
                df.drop(['id'], axis=1, inplace=True)

        # CRITICAL part of the day : Ensure both datasets have the same columns
        print(f"Checking column consistency...")
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)

        # filling the missing ones :))
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols

        if missing_in_test:
            print(f"Adding missing columns to test: {missing_in_test}")
            for col in missing_in_test:
                if X_train[col].dtype == 'object':
                    X_test[col] = 'missing'
                else:
                    X_test[col] = 0

        if missing_in_train:
            print(f" Adding missing columns to train: {missing_in_train}")
            for col in missing_in_train:
                if X_test[col].dtype == 'object':
                    X_train[col] = 'missing'
                else:
                    X_train[col] = 0

        # Reorder columns to match the vibe
        common_cols = sorted(list(train_cols.union(test_cols)))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        print(f"Aligned datasets: {X_train.shape[1]} columns")

        # Identify column types AFTER alignment
        categorical_cols = []
        numerical_cols = []

        for col in X_train.columns:
            # Check if column is categorical
            if (X_train[col].dtype == 'object' or
                col.endswith('_category') or
                col in ['string_id', 'error_code', 'installation_type']):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        print(f"Numerical columns ({len(numerical_cols)}): {len(numerical_cols)} columns")

        # Handle categorical variables with robust encoding
        label_encoders = {}
        for col in categorical_cols:
            try:
                print(f"Encoding {col}...")
                le = LabelEncoder()

                # Combine train and test for consistent encoding
                train_vals = X_train[col].astype(str).fillna('missing')
                test_vals = X_test[col].astype(str).fillna('missing')
                combined = pd.concat([train_vals, test_vals], axis=0)

                # Fit encoder
                le.fit(combined)

                # Transform both datasets
                X_train[col] = le.transform(train_vals)
                X_test[col] = le.transform(test_vals)
                label_encoders[col] = le

                print(f"Encoded {col}: {len(le.classes_)} unique values")

            except Exception as e:
                print(f"Failed to encode {col}: {e}")
                # Remove problematic column
                X_train.drop([col], axis=1, inplace=True)
                X_test.drop([col], axis=1, inplace=True)

        # Handle missing values with simple imputation 
        print("Handling missing values...")

        # Separate numeric and categorical for different imputation strategies
        final_numeric_cols = [col for col in X_train.columns if col not in categorical_cols or col in label_encoders]

        if final_numeric_cols:
            # Use median imputation for numeric
            numeric_imputer = SimpleImputer(strategy='median')

            X_train_numeric = X_train[final_numeric_cols].apply(pd.to_numeric, errors='coerce')
            X_test_numeric = X_test[final_numeric_cols].apply(pd.to_numeric, errors='coerce')

            X_train_imputed = pd.DataFrame(
                numeric_imputer.fit_transform(X_train_numeric),
                columns=final_numeric_cols,
                index=X_train.index
            )

            X_test_imputed = pd.DataFrame(
                numeric_imputer.transform(X_test_numeric),
                columns=final_numeric_cols,
                index=X_test.index
            )

            print(f"Imputed {len(final_numeric_cols)} numeric columns")
        else:
            print("No numeric columns to impute")
            X_train_imputed = X_train.copy()
            X_test_imputed = X_test.copy()

        # Final validation
        print(f"Final shapes - Train: {X_train_imputed.shape}, Test: {X_test_imputed.shape}")
        print(f"Column alignment: {list(X_train_imputed.columns) == list(X_test_imputed.columns)}")

        # Handle target variable outliers if present
        if y_train is not None and len(y_train) > 10:
            try:
                # Convert target to numeric
                y_train = pd.to_numeric(y_train, errors='coerce')

                # Remove rows where target is NaN
                valid_target = y_train.notna()
                if valid_target.sum() < len(y_train):
                    print(f"Removing {(~valid_target).sum()} rows with invalid target values")
                    X_train_imputed = X_train_imputed[valid_target]
                    y_train = y_train[valid_target]
                    train_ids = train_ids[valid_target]

                # Remove extreme outliers (beyond 3 standard deviations)
                if len(y_train) > 20:  # Only if we have enough data
                    z_scores = np.abs((y_train - y_train.mean()) / y_train.std())
                    outlier_mask = z_scores < 3

                    outliers_removed = (~outlier_mask).sum()
                    if outliers_removed > 0 and outlier_mask.sum() > len(y_train) * 0.1:
                        X_train_imputed = X_train_imputed[outlier_mask]
                        y_train = y_train[outlier_mask]
                        train_ids = train_ids[outlier_mask]
                        print(f"🔧 Removed {outliers_removed} target outliers")

            except Exception as e:
                print(f"Target outlier removal failed: {e}")

        print(f"Preprocessing complete!")
        return X_train_imputed, X_test_imputed, y_train, train_ids, test_ids, label_encoders

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# STEP 4: MODEL BUILDING  ( i am model builder now )


def build_models():
    """Build multiple models with conservative parameters"""

    models = {}

    # Always include Random Forest (most reliable)
    models['random_forest'] = RandomForestRegressor(
        n_estimators=50,  # Reduced for speed
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Add MLPRegressor (Neural Network)
    models['mlp'] = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True
    )

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        models['catboost'] = CatBoostRegressor(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )

    print(f" Built {len(models)} models: {list(models.keys())}")
    return models

def create_stratification_labels(y, n_bins=10):
    """Create stratification labels for continuous target variable"""
    # Create bins for stratification
    bins = np.linspace(y.min(), y.max(), n_bins + 1)
    # Assign each value to a bin
    labels = np.digitize(y, bins) - 1
    return labels

def evaluate_models(models, X_train, y_train):
    """Evaluate models using stratified k-fold cross validation"""

    results = {}
    n_splits = 5  # Number of folds for cross validation

    print("=== MODEL EVALUATION ===")

    # Create stratification labels for continuous target
    strat_labels = create_stratification_labels(y_train)

    # Initialize stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            print(f"\nEvaluating {name}...")

            # Initialize metrics lists for each fold
            mae_scores = []
            rmse_scores = []
            r2_scores = []

            # Perform stratified k-fold cross validation
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, strat_labels), 1):
                # Split data for this fold
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                # Train model
                model.fit(X_fold_train, y_fold_train)
                # Make predictions
                pred = model.predict(X_fold_val)

                # Calculate metrics for this fold
                mae = mean_absolute_error(y_fold_val, pred)
                rmse = np.sqrt(mean_squared_error(y_fold_val, pred))
                r2 = r2_score(y_fold_val, pred) if len(set(y_fold_val)) > 1 else 0

                mae_scores.append(mae)
                rmse_scores.append(rmse)
                r2_scores.append(r2)

                print(f"  Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

            # Calculate average metrics across all folds
            avg_mae = np.mean(mae_scores)
            avg_rmse = np.mean(rmse_scores)
            avg_r2 = np.mean(r2_scores)

            results[name] = {
                'mae': avg_mae,
                'rmse': avg_rmse,
                'r2': avg_r2,
                'model': model
            }

            print(f"{name} - Average MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, R²: {avg_r2:.4f}")

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue

    if not results:
        print(" No models could be evaluated!")
        raise Exception("All model evaluations failed")

    return results


# STEP 5: Main pipeline (assembling the parts)


def main_pipeline(train_path, test_path):
    """Main execution pipeline with comprehensive error handling"""

    try:
        print(" Starting Solar Panel ML Pipeline...")

        # Step 1: Load data
        train_df, test_df = load_and_explore_data(train_path, test_path)

        # Step 2: Feature engineering
        print("\n" + "="*50)
        train_df_eng = create_solar_features(train_df)
        test_df_eng = create_solar_features(test_df)

        # Step 3: Preprocessing
        print("\n" + "="*50)
        X_train, X_test, y_train, train_ids, test_ids, encoders = preprocess_data(train_df_eng, test_df_eng)

        if y_train is None:
            print(" No target variable found!")
            return None, None, None

        # Step 4: Build and evaluate models
        print("\n" + "="*50)
        models = build_models()
        results = evaluate_models(models, X_train, y_train)

        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        best_model = results[best_model_name]['model']

        print(f"\nBest model: {best_model_name}")
        print(f"Best MAE: {results[best_model_name]['mae']:.4f}")

        # Step 5: Final training and prediction
        print("\nMaking final predictions...")
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        # Step 6: Create submission
        # Convert test IDs to integers
        test_ids = test_ids.astype(int)
        
        # Create a complete range of IDs first
        all_test_ids = pd.Series(range(len(test_df)))
        
        # Create submission with original predictions
        submission = pd.DataFrame({
            'id': test_ids,
            'efficiency': predictions
        })
        
        # Create a complete submission with all IDs
        complete_submission = pd.DataFrame({
            'id': all_test_ids,
            'efficiency': np.nan  # Initialize with NaN
        })
        
        # Update with actual predictions where we have them
        for idx, row in submission.iterrows():
            complete_submission.loc[complete_submission['id'] == row['id'], 'efficiency'] = row['efficiency']
        
        # Fill any remaining NaN values with the mean prediction
        mean_prediction = submission['efficiency'].mean()
        complete_submission['efficiency'] = complete_submission['efficiency'].fillna(mean_prediction)

        # Save submission
        complete_submission.to_csv('submission.csv', index=False)

        print(f"\n SUCCESS!")
        print(f"Submission saved: submission.csv")
        print(f"Shape: {complete_submission.shape}")
        print(f"Predictions range: {complete_submission['efficiency'].min():.4f} to {complete_submission['efficiency'].max():.4f}")
        print(f"Test IDs range: {complete_submission['id'].min()} to {complete_submission['id'].max()}")

        # Feature importance
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Features:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

        return complete_submission, best_model, feature_importance

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# EXECUTION   (time to test)


if __name__ == "__main__":
    print("Starting Bulletproof Solar Panel ML Pipeline...")

    # Set up proper file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)  # Go up one level to the root directory
    
    train_path = os.path.join(root_dir, 'train.csv')
    test_path = os.path.join(root_dir, 'test.csv')

    # Check if files exist
    for filepath in [train_path, test_path]:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            print(f"Please ensure the dataset files are in the correct location: {root_dir}")
            exit(1)

    # Run pipeline
    result = main_pipeline(train_path, test_path)

    if result[0] is not None:
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("Check 'submission.csv' for your results")
    else:
        print("\nPIPELINE FAILED")
        print("Check the error messages above for debugging info")

#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
import sys
import os

warnings.filterwarnings('ignore')


def load_and_preprocess(filepath):
    """Load and preprocess the evolution dataset."""
    print(f"\n📂 Loading dataset: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Total records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Create derived features
    df['total_independent'] = df['gcid1_independent_change_count'] + df['gcid2_independent_change_count']
    df['total_changes'] = df['co_change_count'] + df['total_independent']
    
    # Filter out rows with no changes (shouldn't exist but just in case)
    df = df[df['total_changes'] > 0].reset_index(drop=True)
    
    # Create label: 1 = independent evolution, 0 = co-evolution (dependent)
    # Independent if more independent changes than co-changes
    df['label'] = (df['total_independent'] > df['co_change_count']).astype(int)
    
    # Create additional features
    df['independent_ratio'] = df['total_independent'] / df['total_changes']
    df['size_diff'] = abs(df['nlines1'] - df['nlines2'])
    df['avg_size'] = (df['nlines1'] + df['nlines2']) / 2
    df['same_author'] = (df['author1'] == df['author2']).astype(int)
    df['same_file'] = (df['depth'] == 0).astype(int)
    
    # Method-based features (new columns from GlobalCloneInfo)
    if 'method_name1' in df.columns and 'method_name2' in df.columns:
        # Check if both clones are in the same method
        df['same_method'] = ((df['method_name1'] == df['method_name2']) & 
                             (df['method_name1'] != 'NULL') & 
                             (df['method_name1'].notna())).astype(int)
        # Check if method info is available for both clones
        df['method_available'] = ((df['method_name1'] != 'NULL') & 
                                   (df['method_name2'] != 'NULL') &
                                   (df['method_name1'].notna()) & 
                                   (df['method_name2'].notna())).astype(int)
        print(f"   Method info available: {df['method_available'].sum()} records ({df['method_available'].mean()*100:.1f}%)")
        print(f"   Same method pairs: {df['same_method'].sum()} records ({df['same_method'].mean()*100:.1f}%)")
    else:
        df['same_method'] = 0
        df['method_available'] = 0
        print(f"   ⚠️  Method columns not found - method features set to 0")
    
    # Encode clone_type if it's categorical (handle both 'object' and 'str' dtypes)
    if df['clone_type'].dtype == 'object' or df['clone_type'].dtype.name == 'str':
        le = LabelEncoder()
        df['clone_type_encoded'] = le.fit_transform(df['clone_type'])
        print(f"   Clone types: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        df['clone_type_encoded'] = df['clone_type']
    
    print(f"\n📊 Label Distribution:")
    print(f"   Independent (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
    print(f"   Co-evolution (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    
    return df


def prepare_features(df):
    """Prepare feature matrix and target vector."""
    features = [
        'depth',
        'similarity', 
        'nlines1', 
        'nlines2',
        'class_size',
        'clone_type_encoded',
        'size_diff',
        'avg_size',
        'same_author',
        'same_file',
        'is_spcp',  # SPCP indicator from Java export
        'same_method',  # Whether both clones are in the same method
        'method_available'  # Whether method info is available for both clones
    ]
    
    # Check which features exist
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Missing features (will be skipped): {missing_features}")
    
    print(f"\n🔧 Features used: {available_features}")
    
    X = df[available_features].copy()
    y = df['label'].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    return X, y, available_features


def train_and_evaluate(X, y, feature_names, df=None):
    """Train multiple models and compare performance.
    
    Uses TEMPORAL SPLIT by revision to avoid data leakage:
    - Earlier revisions → Training set
    - Later revisions → Test set
    """
    print("\n" + "="*60)
    print("🚀 TRAINING MACHINE LEARNING MODELS")
    print("="*60)
    
    # Use temporal split by revision (critical for avoiding data leakage)
    if df is not None and 'Revision' in df.columns:
        print("\n   ⏱️  Using TEMPORAL SPLIT by revision (no data leakage)")
        
        # Get unique revisions sorted
        revisions = sorted(df['Revision'].unique())
        total_revisions = len(revisions)
        
        # Use ~70% of revisions for training, ~30% for testing
        split_idx = int(total_revisions * 0.70)
        train_revisions = set(revisions[:split_idx])
        test_revisions = set(revisions[split_idx:])
        
        # Create masks based on revision
        train_mask = df['Revision'].isin(train_revisions)
        test_mask = df['Revision'].isin(test_revisions)
        
        # Apply masks and reset index for sklearn compatibility
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        
        print(f"   Training revisions: {min(train_revisions)} - {max(train_revisions)} ({len(train_revisions)} revisions)")
        print(f"   Testing revisions:  {min(test_revisions)} - {max(test_revisions)} ({len(test_revisions)} revisions)")
    else:
        # Fallback to random split if no revision info (with warning)
        print("\n   ⚠️  WARNING: No 'Revision' column found - using random split (may cause data leakage)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.35, random_state=42, stratify=y
        )
    
    print(f"\n   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=40, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=40),
        'Logistic Regression': LogisticRegression(random_state=40, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=40)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"📈 Training: {name}")
        print('─'*50)
        
        # Use scaled data for Logistic Regression
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC-ROC:  {auc:.4f}")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Co-evolution', 'Independent']))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model} (Accuracy: {best_accuracy:.4f})")
    print("="*60)
    
    # Feature importance for tree-based models
    print("\n📊 FEATURE IMPORTANCE (Random Forest):")
    print("─"*40)
    rf_model = results['Random Forest']['model']
    importances = list(zip(feature_names, rf_model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"   {feat:25s} {imp:.4f} {bar}")
    
    return results, X_test, y_test, scaler


def analyze_decision_rules(df, feature_names):
    """Analyze what makes clones evolve independently."""
    print("\n" + "="*60)
    print("🔍 EVOLUTION PATTERN ANALYSIS")
    print("="*60)
    
    independent = df[df['label'] == 1]
    coevolving = df[df['label'] == 0]
    
    print("\n📈 Average Feature Values:")
    print("─"*50)
    print(f"{'Feature':<25} {'Independent':>12} {'Co-evolving':>12} {'Diff':>10}")
    print("─"*50)
    
    for feat in ['depth', 'similarity', 'nlines1', 'nlines2', 'class_size', 'same_method', 'method_available']:
        if feat in df.columns:
            ind_mean = independent[feat].mean()
            co_mean = coevolving[feat].mean()
            diff = ind_mean - co_mean
            print(f"{feat:<25} {ind_mean:>12.2f} {co_mean:>12.2f} {diff:>+10.2f}")
    
    # Depth analysis
    print("\n📐 Depth Distribution:")
    print("─"*40)
    for d in sorted(df['depth'].unique())[:6]:
        total = len(df[df['depth'] == d])
        ind = len(df[(df['depth'] == d) & (df['label'] == 1)])
        if total > 0:
            pct = ind / total * 100
            print(f"   Depth {d}: {pct:5.1f}% independent  (n={total})")
    
    # Method analysis (new)
    if 'same_method' in df.columns and df['same_method'].sum() > 0:
        print("\n🔬 Method-Based Analysis:")
        print("─"*40)
        
        # Same method vs different method
        same_method_df = df[df['same_method'] == 1]
        diff_method_df = df[df['same_method'] == 0]
        
        if len(same_method_df) > 0:
            same_method_ind_pct = (same_method_df['label'] == 1).mean() * 100
            print(f"   Same method pairs: {same_method_ind_pct:5.1f}% evolve independently  (n={len(same_method_df)})")
        
        if len(diff_method_df) > 0:
            diff_method_ind_pct = (diff_method_df['label'] == 1).mean() * 100
            print(f"   Diff method pairs: {diff_method_ind_pct:5.1f}% evolve independently  (n={len(diff_method_df)})")
        
        # Method info available vs not
        if 'method_available' in df.columns:
            has_method = df[df['method_available'] == 1]
            no_method = df[df['method_available'] == 0]
            
            if len(has_method) > 0:
                has_method_ind_pct = (has_method['label'] == 1).mean() * 100
                print(f"   With method info:  {has_method_ind_pct:5.1f}% evolve independently  (n={len(has_method)})")
            
            if len(no_method) > 0:
                no_method_ind_pct = (no_method['label'] == 1).mean() * 100
                print(f"   No method info:    {no_method_ind_pct:5.1f}% evolve independently  (n={len(no_method)})")


def predict_new_clone_pair(model, scaler, feature_names, clone_pair_data):
    """Predict evolution type for a new clone pair."""
    # This function can be used for real-time prediction
    df = pd.DataFrame([clone_pair_data])
    X = df[feature_names].fillna(0)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        'prediction': 'Independent' if prediction == 1 else 'Co-evolution',
        'confidence': max(probability),
        'independent_probability': probability[1],
        'coevolution_probability': probability[0]
    }


def save_predictions(df, results, output_path):
    """Save predictions to CSV."""
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    # Note: predictions are only for test set, so we can't add to full df
    print(f"\n💾 Model trained. Use predict_new_clone_pair() for new predictions.")
    

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🧬 CLONE EVOLUTION FORECASTING - ML PIPELINE")
    print("="*60)
    
    # Get dataset path
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default path - look for evolution dataset
        possible_paths = [
            'evolution_dataset.csv',
            'WorkFolder/jmol/Datasets/CloneGenealogy/Type3_Block_evolution_dataset.csv',
            'WorkFolder/jmol/Datasets/CloneGenealogy/Type1_Block_evolution_dataset.csv',
        ]
        filepath = None
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break
        
        if filepath is None:
            print("\nError: No dataset found.")
            print("   Usage: python clone_evolution_ml.py <path_to_evolution_dataset.csv>")
            sys.exit(1)
    
    # Load and preprocess
    df = load_and_preprocess(filepath)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train and evaluate (uses temporal split by revision to avoid data leakage)
    results, X_test, y_test, scaler = train_and_evaluate(X, y, feature_names, df)
    
    # Analyze patterns
    analyze_decision_rules(df, feature_names)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
    return df, results, scaler, feature_names


if __name__ == '__main__':
    df, results, scaler, features = main()

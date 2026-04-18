#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, precision_recall_fscore_support, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import sys
import os

warnings.filterwarnings('ignore')


def load_pairs_data(filepath):
    """Load and preprocess the clone pairs CSV."""
    print(f"\n{'='*60}")
    print("📂 LOADING CLONE PAIRS DATA")
    print('='*60)
    print(f"   File: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"   Total pairs: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def engineer_features(df):
    """Create features for ML from clone pairs data."""
    print(f"\n{'='*60}")
    print("🔧 FEATURE ENGINEERING")
    print('='*60)
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # === Target Variable ===
    # A pair has "independent evolution" if total independent changes > co-changes
    data['total_independent'] = data['gcid1_independent_change'] + data['gcid2_independent_change']
    data['total_changes'] = data['co_change'] + data['total_independent']
    
    # Filter pairs with at least some change activity
    data = data[data['total_changes'] > 0].reset_index(drop=True)
    print(f"   Pairs with change activity: {len(data)}")
    
    # Binary label: 1 = Independent evolution tendency, 0 = Co-evolution tendency
    data['label'] = (data['total_independent'] > data['co_change']).astype(int)
    
    # === Derived Features ===
    
    # Evolution ratio features
    data['independent_ratio'] = data['total_independent'] / data['total_changes']
    data['cochange_ratio'] = data['co_change'] / data['total_changes']
    data['gcid1_change_ratio'] = data['gcid1_independent_change'] / data['total_changes']
    data['gcid2_change_ratio'] = data['gcid2_independent_change'] / data['total_changes']
    
    # Change imbalance: how asymmetric are the independent changes
    data['change_imbalance'] = abs(data['gcid1_independent_change'] - data['gcid2_independent_change'])
    data['change_imbalance_ratio'] = data['change_imbalance'] / (data['total_independent'] + 1)
    
    # Lifespan features
    data['lifespan_diff'] = abs(data['clone1_lifespan'] - data['clone2_lifespan'])
    data['avg_lifespan'] = (data['clone1_lifespan'] + data['clone2_lifespan']) / 2
    data['min_lifespan'] = data[['clone1_lifespan', 'clone2_lifespan']].min(axis=1)
    data['max_lifespan'] = data[['clone1_lifespan', 'clone2_lifespan']].max(axis=1)
    
    # File location features
    data['same_file'] = (data['file_distance'] == 0).astype(int)
    data['different_folder'] = (data['file_distance'] > 1).astype(int)
    
    # Author features
    data['gcid1_author_count'] = data['gcid1_authors'].apply(lambda x: len(str(x).split('; ')) if pd.notna(x) else 0)
    data['gcid2_author_count'] = data['gcid2_authors'].apply(lambda x: len(str(x).split('; ')) if pd.notna(x) else 0)
    data['total_unique_authors'] = data.apply(
        lambda row: len(set(str(row['gcid1_authors']).split('; ')) | set(str(row['gcid2_authors']).split('; '))),
        axis=1
    )
    data['author_overlap'] = data.apply(
        lambda row: len(set(str(row['gcid1_authors']).split('; ')) & set(str(row['gcid2_authors']).split('; '))),
        axis=1
    )
    data['author_overlap_ratio'] = data['author_overlap'] / (data['total_unique_authors'] + 1)
    
    # Added revision difference
    data['added_rev_diff'] = abs(data['clone1_addedRev'] - data['clone2_addedRev'])
    data['same_origin'] = (data['added_rev_diff'] == 0).astype(int)
    
    # Stability features (based on change frequency over lifespan)
    data['gcid1_change_frequency'] = (data['gcid1_independent_change'] + data['co_change']) / (data['clone1_lifespan'] + 1)
    data['gcid2_change_frequency'] = (data['gcid2_independent_change'] + data['co_change']) / (data['clone2_lifespan'] + 1)
    
    print(f"\n   Engineered {len([c for c in data.columns if c not in df.columns])} new features")
    
    # Print label distribution
    print(f"\n📊 Label Distribution:")
    print(f"   Independent Evolution (1): {(data['label'] == 1).sum()} ({(data['label'] == 1).mean()*100:.1f}%)")
    print(f"   Co-Evolution (0): {(data['label'] == 0).sum()} ({(data['label'] == 0).mean()*100:.1f}%)")
    
    return data


def prepare_features(df):
    """Prepare feature matrix and target vector.
    
    IMPORTANT: Only include features that would be available BEFORE 
    evolution happens (no data leakage from change counts).
    """
    # Features available for FORECASTING (before changes happen)
    # These are structural/contextual features, not derived from change history
    forecast_features = [
        # Clone similarity and structure
        'similarity',
        'file_distance',
        'same_file',
        'different_folder',
        
        # Lifespan features (proxy for code stability)
        'clone1_lifespan',
        'clone2_lifespan',
        'lifespan_diff',
        'avg_lifespan',
        'min_lifespan',
        'max_lifespan',
        
        # Author diversity features (proxy for maintenance complexity)
        'gcid1_author_count',
        'gcid2_author_count',
        'total_unique_authors',
        'author_overlap',
        'author_overlap_ratio',
        
        # Origin features
        'added_rev_diff',
        'same_origin',
    ]
    
    # Filter to available features
    available = [f for f in forecast_features if f in df.columns]
    missing = [f for f in forecast_features if f not in df.columns]
    
    if missing:
        print(f"\n⚠️  Missing features (skipped): {missing[:5]}...")
    
    print(f"\n🔧 Using {len(available)} features for forecasting (no data leakage)")
    print(f"   Features: {available}")
    
    X = df[available].copy().fillna(0)
    y = df['label'].copy()
    
    return X, y, available


def train_models(X, y, feature_names, df=None):
    """Train and evaluate multiple ML models.
    
    Uses TEMPORAL SPLIT by revision to avoid data leakage:
    - Earlier revisions → Training set
    - Later revisions → Test set
    """
    print(f"\n{'='*60}")
    print("🚀 TRAINING MACHINE LEARNING MODELS")
    print('='*60)
    
    # Use temporal split by revision (critical for avoiding data leakage)
    if df is not None and 'clone1_addedRev' in df.columns:
        print("\n   ⏱️  Using TEMPORAL SPLIT by revision (no data leakage)")
        
        # Use clone1_addedRev as the temporal indicator
        revisions = sorted(df['clone1_addedRev'].unique())
        total_revisions = len(revisions)
        
        # Use ~70% of revisions for training, ~30% for testing
        split_idx = int(total_revisions * 0.70)
        train_revisions = set(revisions[:split_idx])
        test_revisions = set(revisions[split_idx:])
        
        # Create masks based on revision
        train_mask = df['clone1_addedRev'].isin(train_revisions)
        test_mask = df['clone1_addedRev'].isin(test_revisions)
        
        # Apply masks and reset index for sklearn compatibility
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        
        print(f"   Training revisions: {min(train_revisions)} - {max(train_revisions)} ({len(train_revisions)} revisions)")
        print(f"   Testing revisions:  {min(test_revisions)} - {max(test_revisions)} ({len(test_revisions)} revisions)")
    else:
        # Fallback to random split if no revision info (with warning)
        print("\n   ⚠️  WARNING: No revision column found - using random split (may cause data leakage)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    
    print(f"\n   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5, 
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, C=1.0
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=10, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50, learning_rate=0.5, random_state=42
        ),
    }
    
    results = {}
    best_model_name = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"📈 {name}")
        print('─'*50)
        
        # Use scaled data for Logistic Regression
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc:.4f}")
        print(f"   CV Score:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print('='*60)
    
    # Feature Importance
    print(f"\n📊 FEATURE IMPORTANCE (Random Forest):")
    print('─'*50)
    rf = results['Random Forest']['model']
    importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:15]:  # Top 15
        bar = '█' * int(imp * 40)
        print(f"   {feat:30s} {imp:.4f} {bar}")
    
    return results, X_test, y_test, scaler, X_train, y_train


def analyze_patterns(df):
    """Analyze evolution patterns in the data."""
    print(f"\n{'='*60}")
    print("🔍 EVOLUTION PATTERN ANALYSIS")
    print('='*60)
    
    independent = df[df['label'] == 1]
    coevolving = df[df['label'] == 0]
    
    print(f"\n📈 Key Differences (Independent vs Co-evolving):")
    print('─'*60)
    print(f"{'Feature':<30} {'Independent':>12} {'Co-evolving':>12} {'Diff':>10}")
    print('─'*60)
    
    compare_features = [
        'similarity', 'file_distance', 'avg_lifespan', 
        'author_overlap_ratio', 'total_unique_authors',
        'gcid1_author_count', 'gcid2_author_count'
    ]
    
    for feat in compare_features:
        if feat in df.columns:
            ind_mean = independent[feat].mean()
            co_mean = coevolving[feat].mean()
            diff = ind_mean - co_mean
            print(f"{feat:<30} {ind_mean:>12.2f} {co_mean:>12.2f} {diff:>+10.2f}")
    
    # File distance analysis
    print(f"\n📐 File Distance vs Independence:")
    print('─'*40)
    for d in sorted(df['file_distance'].unique())[:6]:
        subset = df[df['file_distance'] == d]
        ind_pct = subset['label'].mean() * 100
        print(f"   Distance {d}: {ind_pct:5.1f}% independent (n={len(subset)})")
    
    # Similarity analysis
    print(f"\n🔗 Similarity vs Independence:")
    print('─'*40)
    sim_bins = [(0, 70), (70, 80), (80, 90), (90, 100)]
    for low, high in sim_bins:
        subset = df[(df['similarity'] >= low) & (df['similarity'] < high)]
        if len(subset) > 0:
            ind_pct = subset['label'].mean() * 100
            print(f"   Similarity {low}-{high}%: {ind_pct:5.1f}% independent (n={len(subset)})")
    
    # Author overlap analysis
    print(f"\n👥 Author Overlap vs Independence:")
    print('─'*40)
    for overlap in range(min(5, int(df['author_overlap'].max()) + 1)):
        subset = df[df['author_overlap'] == overlap]
        if len(subset) > 0:
            ind_pct = subset['label'].mean() * 100
            print(f"   {overlap} common authors: {ind_pct:5.1f}% independent (n={len(subset)})")


def predict_independence_risk(model, scaler, feature_names, new_pair):
    """Predict independence risk for a new clone pair."""
    df = pd.DataFrame([new_pair])
    X = df[feature_names].fillna(0)
    X_scaled = scaler.transform(X)
    
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    return {
        'prediction': 'Independent Evolution' if pred == 1 else 'Co-Evolution',
        'confidence': max(proba) * 100,
        'independence_probability': proba[1] * 100,
        'coevolution_probability': proba[0] * 100
    }


def export_predictions(df, results, output_path):
    """Export predictions to CSV for further analysis."""
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    # Get features used
    feature_cols = [c for c in df.columns if c not in ['label', 'gcid1_authors', 'gcid2_authors', 
                                                        'filepath1', 'filepath2', 'cloneType', 
                                                        'granularity', 'clone1_deletedRev', 'clone2_deletedRev']]
    
    print(f"\n💾 Predictions can be made using the trained {best_model_name} model")
    print(f"   Use predict_independence_risk() for new clone pairs")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🧬 CLONE PAIRS INDEPENDENT EVOLUTION FORECASTING")
    print("="*60)
    
    # Get dataset path
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Try default paths
        possible_paths = [
            'WorkFolder/dnsjava/Datasets/CloneGenealogy/Type3_Block_pairs.csv',
            'WorkFolder/dnsjava/Datasets/CloneGenealogy/Type2_Block_pairs.csv',
            'WorkFolder/dnsjava/Datasets/CloneGenealogy/Type1_Block_pairs.csv',
            'Type3_Block_pairs.csv',
        ]
        filepath = None
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break
        
        if filepath is None:
            print("\n❌ Error: No pairs CSV found.")
            print("   Usage: python clone_pairs_forecasting.py <path_to_pairs.csv>")
            sys.exit(1)
    
    # Load data
    df = load_pairs_data(filepath)
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train models (uses temporal split by clone1_addedRev to avoid data leakage)
    results, X_test, y_test, scaler, X_train, y_train = train_models(X, y, feature_names, df)
    
    # Analyze patterns
    analyze_patterns(df)
    
    # Summary
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print("\n📋 Summary:")
    print(f"   - Total clone pairs analyzed: {len(df)}")
    print(f"   - Features engineered: {len(feature_names)}")
    print(f"   - Best model: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]}")
    print(f"   - Best accuracy: {max(r['accuracy'] for r in results.values()):.4f}")
    
    return df, results, scaler, feature_names


if __name__ == '__main__':
    df, results, scaler, features = main()

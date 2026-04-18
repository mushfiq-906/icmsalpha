#!/usr/bin/env python3
"""
Combined Clone Evolution ML Pipeline

This script merges multiple CSV datasets from CloneGenealogy folder for enhanced
clone evolution forecasting. It enriches the primary evolution dataset with 
features from clone-level, pair-level, SPCP, and class group datasets.

Usage:
    python clone_evolution_combined_ml.py <path_to_CloneGenealogy_folder>
    
Example:
    python clone_evolution_combined_ml.py WorkFolder/jmol/Datasets/CloneGenealogy/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings
import sys
import os

warnings.filterwarnings('ignore')


def load_all_datasets(folder_path):
    """Load all CSV files from the CloneGenealogy folder."""
    print(f"\n{'='*70}")
    print("📂 LOADING ALL DATASETS")
    print('='*70)
    
    datasets = {}
    
    # Find and load all relevant CSVs
    csv_patterns = {
        'evolution': '_evolution_dataset.csv',
        'pairs': '_pairs.csv', 
        'clones': None,  # Will match Type*_Block.csv (not pairs or evolution)
        'classGroups': '_classGroups.csv',
        'spcp': 'SPCP_'
    }
    
    # List files in folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"   Found {len(files)} CSV files")
    
    for f in files:
        filepath = os.path.join(folder_path, f)
        
        if '_evolution_dataset.csv' in f:
            datasets['evolution'] = pd.read_csv(filepath)
            print(f"   ✓ Evolution dataset: {len(datasets['evolution'])} rows")
        elif '_pairs.csv' in f:
            datasets['pairs'] = pd.read_csv(filepath)
            print(f"   ✓ Pairs dataset: {len(datasets['pairs'])} rows")
        elif '_classGroups.csv' in f:
            datasets['classGroups'] = pd.read_csv(filepath)
            print(f"   ✓ Class groups: {len(datasets['classGroups'])} rows")
        elif f.startswith('SPCP_'):
            datasets['spcp'] = pd.read_csv(filepath)
            print(f"   ✓ SPCP dataset: {len(datasets['spcp'])} rows")
        elif 'Block.csv' in f and 'pairs' not in f and 'evolution' not in f and 'classGroups' not in f:
            datasets['clones'] = pd.read_csv(filepath)
            print(f"   ✓ Clones dataset: {len(datasets['clones'])} rows")
    
    if 'evolution' not in datasets:
        raise FileNotFoundError("Primary evolution dataset not found!")
    
    return datasets


def merge_datasets(datasets):
    """Merge all datasets into enriched evolution dataset."""
    print(f"\n{'='*70}")
    print("🔗 MERGING DATASETS")
    print('='*70)
    
    df = datasets['evolution'].copy()
    original_cols = len(df.columns)
    print(f"   Base evolution dataset: {len(df)} rows, {original_cols} columns")
    
    # === Merge Clone-level features (stabilityIndex, changeProneness) ===
    if 'clones' in datasets:
        clones = datasets['clones']
        
        # Features to extract per clone
        clone_features = ['globalCloneId', 'stabilityIndex', 'changeProneness', 
                         'independentEvolutionScore', 'totalChanges', 'totalUnchanged']
        available_clone_features = [c for c in clone_features if c in clones.columns]
        
        if 'globalCloneId' in available_clone_features:
            clone_data = clones[available_clone_features].copy()
            
            # Merge for gcid1
            clone_data_1 = clone_data.rename(columns={
                c: f'{c}_clone1' if c != 'globalCloneId' else c 
                for c in clone_data.columns
            })
            df = df.merge(clone_data_1, left_on='gcid1', right_on='globalCloneId', how='left')
            df.drop(columns=['globalCloneId'], errors='ignore', inplace=True)
            
            # Merge for gcid2
            clone_data_2 = clone_data.rename(columns={
                c: f'{c}_clone2' if c != 'globalCloneId' else c 
                for c in clone_data.columns
            })
            df = df.merge(clone_data_2, left_on='gcid2', right_on='globalCloneId', how='left')
            df.drop(columns=['globalCloneId'], errors='ignore', inplace=True)
            
            print(f"   ✓ Added clone-level features (stabilityIndex, changeProneness, etc.)")
    
    # === Merge Pair-level features (lifespan, file_distance, authors) ===
    if 'pairs' in datasets:
        pairs = datasets['pairs']
        
        # Features from pairs
        pair_features = ['clone1_gcid', 'clone2_gcid', 'clone1_lifespan', 'clone2_lifespan',
                        'file_distance', 'gcid1_authors', 'gcid2_authors']
        available_pair_features = [c for c in pair_features if c in pairs.columns]
        
        if 'clone1_gcid' in available_pair_features and 'clone2_gcid' in available_pair_features:
            pair_data = pairs[available_pair_features].drop_duplicates(
                subset=['clone1_gcid', 'clone2_gcid']
            )
            
            # Merge on gcid pair
            df = df.merge(pair_data, 
                         left_on=['gcid1', 'gcid2'], 
                         right_on=['clone1_gcid', 'clone2_gcid'],
                         how='left')
            df.drop(columns=['clone1_gcid', 'clone2_gcid'], errors='ignore', inplace=True)
            
            print(f"   ✓ Added pair-level features (lifespan, file_distance, authors)")
    
    # === Merge SPCP features (couplingStrength, dominantChangeCategory) ===
    if 'spcp' in datasets:
        spcp = datasets['spcp']
        
        spcp_features = ['gcid1', 'gcid2', 'couplingStrength', 'latePropagationCount', 
                        'dominantChangeCategory', 'no_of_revision_paired']
        available_spcp_features = [c for c in spcp_features if c in spcp.columns]
        
        if 'gcid1' in available_spcp_features and 'gcid2' in available_spcp_features:
            spcp_data = spcp[available_spcp_features].copy()
            spcp_data = spcp_data.rename(columns={
                'couplingStrength': 'spcp_coupling_strength',
                'latePropagationCount': 'spcp_late_propagation',
                'dominantChangeCategory': 'spcp_dominant_category',
                'no_of_revision_paired': 'spcp_revision_count'
            })
            
            # Merge SPCP data
            df = df.merge(spcp_data, on=['gcid1', 'gcid2'], how='left')
            print(f"   ✓ Added SPCP features (couplingStrength, dominantChangeCategory)")
    
    # === Merge Class group size ===
    if 'classGroups' in datasets:
        class_groups = datasets['classGroups']
        
        if 'classId' in class_groups.columns and 'clone_gcid_list' in class_groups.columns:
            # Parse the list and count members
            class_groups['class_group_size'] = class_groups['clone_gcid_list'].apply(
                lambda x: len(str(x).replace('[', '').replace(']', '').split(',')) if pd.notna(x) else 0
            )
            class_size_map = class_groups[['classId', 'class_group_size']].copy()
            
            # Merge on classId
            if 'classid' in df.columns:
                df = df.merge(class_size_map, left_on='classid', right_on='classId', how='left')
                df.drop(columns=['classId'], errors='ignore', inplace=True)
            
            print(f"   ✓ Added class group size feature")
    
    new_cols = len(df.columns)
    print(f"\n   📊 Enriched dataset: {len(df)} rows, {new_cols} columns (+{new_cols - original_cols} features)")
    
    return df


def engineer_combined_features(df):
    """Create derived features from merged data."""
    print(f"\n{'='*70}")
    print("🔧 FEATURE ENGINEERING")
    print('='*70)
    
    # Reset index after any potential filtering
    df = df.reset_index(drop=True)
    
    # === Basic derived features ===
    df['total_independent'] = df['gcid1_independent_change_count'] + df['gcid2_independent_change_count']
    df['total_changes'] = df['co_change_count'] + df['total_independent']
    
    # Filter out rows with no changes
    df = df[df['total_changes'] > 0].reset_index(drop=True)
    
    # Label: 1 = independent evolution, 0 = co-evolution
    df['label'] = (df['total_independent'] > df['co_change_count']).astype(int)
    
    # === Evolution ratio features ===
    df['independent_ratio'] = df['total_independent'] / df['total_changes']
    df['size_diff'] = abs(df['nlines1'] - df['nlines2'])
    df['avg_size'] = (df['nlines1'] + df['nlines2']) / 2
    df['same_author'] = (df['author1'] == df['author2']).astype(int)
    df['same_file'] = (df['depth'] == 0).astype(int)
    
    # === Method features ===
    if 'method_name1' in df.columns and 'method_name2' in df.columns:
        df['same_method'] = ((df['method_name1'] == df['method_name2']) & 
                             (df['method_name1'] != 'NULL') & 
                             (df['method_name1'].notna())).astype(int)
        df['method_available'] = ((df['method_name1'] != 'NULL') & 
                                   (df['method_name2'] != 'NULL') &
                                   (df['method_name1'].notna()) & 
                                   (df['method_name2'].notna())).astype(int)
    else:
        df['same_method'] = 0
        df['method_available'] = 0
    
    # === Clone stability features (from merged clone data) ===
    if 'stabilityIndex_clone1' in df.columns:
        df['avg_stability'] = (df['stabilityIndex_clone1'].fillna(0) + 
                               df['stabilityIndex_clone2'].fillna(0)) / 2
        df['stability_diff'] = abs(df['stabilityIndex_clone1'].fillna(0) - 
                                   df['stabilityIndex_clone2'].fillna(0))
        print("   ✓ Created stability features")
    
    if 'changeProneness_clone1' in df.columns:
        df['avg_change_proneness'] = (df['changeProneness_clone1'].fillna(0) + 
                                      df['changeProneness_clone2'].fillna(0)) / 2
        df['change_proneness_diff'] = abs(df['changeProneness_clone1'].fillna(0) - 
                                          df['changeProneness_clone2'].fillna(0))
        print("   ✓ Created change proneness features")
    
    if 'independentEvolutionScore_clone1' in df.columns:
        df['avg_ind_evolution_score'] = (df['independentEvolutionScore_clone1'].fillna(0) + 
                                         df['independentEvolutionScore_clone2'].fillna(0)) / 2
        print("   ✓ Created independent evolution score features")
    
    # === Lifespan features (from pairs data) ===
    if 'clone1_lifespan' in df.columns:
        df['lifespan_diff'] = abs(df['clone1_lifespan'].fillna(0) - df['clone2_lifespan'].fillna(0))
        df['avg_lifespan'] = (df['clone1_lifespan'].fillna(0) + df['clone2_lifespan'].fillna(0)) / 2
        df['min_lifespan'] = df[['clone1_lifespan', 'clone2_lifespan']].min(axis=1)
        print("   ✓ Created lifespan features")
    
    # === Author diversity features ===
    if 'gcid1_authors' in df.columns:
        df['author1_count'] = df['gcid1_authors'].apply(
            lambda x: len(str(x).split('; ')) if pd.notna(x) and x != 'nan' else 0
        )
        df['author2_count'] = df['gcid2_authors'].apply(
            lambda x: len(str(x).split('; ')) if pd.notna(x) and x != 'nan' else 0
        )
        df['total_authors'] = df['author1_count'] + df['author2_count']
        print("   ✓ Created author diversity features")
    
    # === SPCP features ===
    if 'spcp_coupling_strength' in df.columns:
        df['has_spcp'] = df['spcp_coupling_strength'].notna().astype(int)
        df['spcp_coupling_strength'] = df['spcp_coupling_strength'].fillna(0)
        df['spcp_late_propagation'] = df['spcp_late_propagation'].fillna(0)
        print("   ✓ Created SPCP features")
    
    # Encode categorical columns
    if 'clone_type' in df.columns:
        if df['clone_type'].dtype == 'object' or df['clone_type'].dtype.name == 'str':
            le = LabelEncoder()
            df['clone_type_encoded'] = le.fit_transform(df['clone_type'].astype(str))
        else:
            df['clone_type_encoded'] = df['clone_type']
    
    if 'spcp_dominant_category' in df.columns:
        df['spcp_dominant_category'] = df['spcp_dominant_category'].fillna('NONE')
        le_spcp = LabelEncoder()
        df['spcp_category_encoded'] = le_spcp.fit_transform(df['spcp_dominant_category'].astype(str))
        print(f"   SPCP categories: {dict(zip(le_spcp.classes_, range(len(le_spcp.classes_))))}")
    
    # Print label distribution
    print(f"\n📊 Label Distribution:")
    print(f"   Independent (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
    print(f"   Co-evolution (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    
    return df


def prepare_combined_features(df):
    """Prepare the full feature matrix."""
    
    # Define all possible features (will filter to available)
    all_features = [
        # Original features
        'depth', 'similarity', 'nlines1', 'nlines2', 'class_size',
        'clone_type_encoded', 'size_diff', 'avg_size', 'same_author', 
        'same_file', 'is_spcp', 'same_method', 'method_available',
        
        # Clone stability features
        'stabilityIndex_clone1', 'stabilityIndex_clone2', 'avg_stability', 'stability_diff',
        'changeProneness_clone1', 'changeProneness_clone2', 'avg_change_proneness', 'change_proneness_diff',
        'independentEvolutionScore_clone1', 'independentEvolutionScore_clone2', 'avg_ind_evolution_score',
        
        # Lifespan features  
        'clone1_lifespan', 'clone2_lifespan', 'lifespan_diff', 'avg_lifespan', 'min_lifespan',
        'file_distance',
        
        # Author features
        'author1_count', 'author2_count', 'total_authors',
        
        # SPCP features
        'has_spcp', 'spcp_coupling_strength', 'spcp_late_propagation', 'spcp_category_encoded',
        
        # Class group features
        'class_group_size',
    ]
    
    # Filter to available
    available = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    
    print(f"\n🔧 FEATURE SELECTION")
    print(f"   Available features: {len(available)}")
    print(f"   Features: {available}")
    if missing:
        print(f"   Missing (skipped): {len(missing)}")
    
    X = df[available].copy().fillna(0)
    y = df['label'].copy()
    
    return X, y, available


def train_combined_models(X, y, feature_names, df):
    """Train models with temporal split."""
    print(f"\n{'='*70}")
    print("🚀 TRAINING MACHINE LEARNING MODELS")
    print('='*70)
    
    # Temporal split by revision
    if 'Revision' in df.columns:
        print("\n   ⏱️  Using TEMPORAL SPLIT by revision (no data leakage)")
        
        revisions = sorted(df['Revision'].unique())
        split_idx = int(len(revisions) * 0.70)
        train_revisions = set(revisions[:split_idx])
        test_revisions = set(revisions[split_idx:])
        
        train_mask = df['Revision'].isin(train_revisions)
        test_mask = df['Revision'].isin(test_revisions)
        
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        
        print(f"   Training revisions: {min(train_revisions)} - {max(train_revisions)} ({len(train_revisions)} revisions)")
        print(f"   Testing revisions:  {min(test_revisions)} - {max(test_revisions)} ({len(test_revisions)} revisions)")
    else:
        print("\n   ⚠️  WARNING: No Revision column - using random split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"\n   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\n{'─'*60}")
        print(f"📈 Training: {name}")
        print('─'*60)
        
        # Use scaled data for Logistic Regression and Naive Bayes
        if 'Logistic' in name or 'Naive Bayes' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC-ROC:  {auc:.4f}")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Co-evolution', 'Independent']))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_model} (Accuracy: {best_accuracy:.4f})")
    print('='*70)
    
    # Feature importance
    print(f"\n📊 FEATURE IMPORTANCE (Random Forest):")
    print('─'*60)
    rf = results['Random Forest']['model']
    importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:20]:
        bar = '█' * int(imp * 50)
        print(f"   {feat:35s} {imp:.4f} {bar}")
    
    return results, X_test, y_test, scaler


def analyze_combined_patterns(df):
    """Analyze evolution patterns with enriched features."""
    print(f"\n{'='*70}")
    print("🔍 ENRICHED PATTERN ANALYSIS")
    print('='*70)
    
    independent = df[df['label'] == 1]
    coevolving = df[df['label'] == 0]
    
    print(f"\n📈 Key Feature Differences:")
    print('─'*60)
    print(f"{'Feature':<35} {'Independent':>12} {'Co-evolving':>12} {'Diff':>10}")
    print('─'*60)
    
    compare_features = [
        'depth', 'similarity', 'avg_stability', 'avg_change_proneness',
        'avg_lifespan', 'file_distance', 'total_authors', 'spcp_coupling_strength'
    ]
    
    for feat in compare_features:
        if feat in df.columns:
            ind_mean = independent[feat].mean()
            co_mean = coevolving[feat].mean()
            diff = ind_mean - co_mean
            print(f"{feat:<35} {ind_mean:>12.2f} {co_mean:>12.2f} {diff:>+10.2f}")
    
    # SPCP Analysis
    if 'has_spcp' in df.columns:
        print(f"\n🔗 SPCP Pairs Analysis:")
        print('─'*40)
        spcp_pairs = df[df['has_spcp'] == 1]
        non_spcp = df[df['has_spcp'] == 0]
        
        if len(spcp_pairs) > 0:
            spcp_ind = (spcp_pairs['label'] == 1).mean() * 100
            print(f"   SPCP pairs:     {spcp_ind:5.1f}% evolve independently (n={len(spcp_pairs)})")
        if len(non_spcp) > 0:
            non_spcp_ind = (non_spcp['label'] == 1).mean() * 100
            print(f"   Non-SPCP pairs: {non_spcp_ind:5.1f}% evolve independently (n={len(non_spcp)})")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("🧬 COMBINED CLONE EVOLUTION FORECASTING - MULTI-DATASET ML PIPELINE")
    print("="*70)
    
    # Get folder path
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # Try default paths
        possible_paths = [
            'WorkFolder/jmol/Datasets/CloneGenealogy/',
            'WorkFolder/dnsjava/Datasets/CloneGenealogy/',
        ]
        folder_path = None
        for p in possible_paths:
            if os.path.exists(p):
                folder_path = p
                break
        
        if folder_path is None:
            print("\n❌ Error: No CloneGenealogy folder found.")
            print("   Usage: python clone_evolution_combined_ml.py <path_to_CloneGenealogy_folder>")
            sys.exit(1)
    
    # Load all datasets
    datasets = load_all_datasets(folder_path)
    
    # Merge datasets
    df = merge_datasets(datasets)
    
    # Engineer features
    df = engineer_combined_features(df)
    
    # Prepare features
    X, y, feature_names = prepare_combined_features(df)
    
    # Train models
    results, X_test, y_test, scaler = train_combined_models(X, y, feature_names, df)
    
    # Analyze patterns
    analyze_combined_patterns(df)
    
    print("\n" + "="*70)
    print("✅ COMBINED ANALYSIS COMPLETE")
    print("="*70)
    
    return df, results, scaler, feature_names


if __name__ == '__main__':
    df, results, scaler, features = main()

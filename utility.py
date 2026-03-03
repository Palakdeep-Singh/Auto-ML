from models_registry import MODEL_REGISTRY
from flask import Flask, render_template, request, redirect, send_file, url_for, jsonify, flash, session, Response
from pandas.api.types import is_numeric_dtype, is_string_dtype
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, json, warnings, math, time
import json
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import warnings
import threading
import time

from state import *

# ----------------------------------------------------------------------
# Helper to get models for a stage (used in zero‑inflated training)
# ----------------------------------------------------------------------
def _get_models_for_stage(form_data, task_group, default_models):
    """Extract valid models for a given task group from form data."""
    models_value = form_data.get("models", [])
    if isinstance(models_value, str):
        models_value = [models_value]
    if not models_value:
        return default_models
    # Keep only models that exist in the registry for the given task
    available = MODEL_REGISTRY.get(task_group, {})
    return [m for m in models_value if m in available]

# ----------------------------------------------------------------------
# Existing helper functions (unchanged, but kept for completeness)
# ----------------------------------------------------------------------
def format_number_filter(value):
    """Jinja2 filter: format numbers with commas and smart decimal places."""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "N/A"
            if value == int(value):
                return f"{int(value):,}"
            if abs(value) >= 1:
                return f"{value:,.4f}"
            return f"{value:.6f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)
    except Exception:
        return str(value)

def process_form_data(form_data):
    """
    Process form data with proper parameter handling for AutoML frontend
    """
    # [DEBUG] print(f"DEBUG: Raw form data received - {list(form_data.keys())}")

    result = {}

    if "models[]" in form_data:
        form_data["models"] = form_data.pop("models[]")

    # Process basic fields
    for key, value_list in form_data.items():
        if key in ['model_mode', 'tuning_mode']:
            result[key] = value_list[0] if value_list else 'auto'
        elif key == 'models':
            # Handle model selection (list of models)
            if value_list:
                result[key] = value_list if isinstance(value_list, list) else [value_list]
                # [DEBUG] print(f"DEBUG: Models selected: {result[key]}")
            else:
                result[key] = []
        else:
            # Handle model parameters (single value expected)
            if value_list and len(value_list) > 0:
                value = value_list[0]

                # Handle different parameter types
                if isinstance(value, str):
                    # Convert based on string content
                    if value.lower() in ["true", "false"]:
                        result[key] = value.lower() == "true"
                    elif value == "None" or value == "null" or value == "":
                        result[key] = None
                    elif value.isdigit():
                        result[key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        result[key] = float(value)
                    else:
                        result[key] = value
                else:
                    result[key] = value

    # Debug output for parameters
    param_keys = [k for k in result.keys() if '__' in k]
    # [DEBUG] print(f"DEBUG: Found {len(param_keys)} parameter keys")
    if param_keys:
        # [DEBUG] print(f"DEBUG: Parameter keys (first 10):")
        for i, k in enumerate(param_keys[:10]):
            pass  # placeholder (debug print removed)
            # [DEBUG] print(f"  {i+1}. {k} = {result[k]} (type: {type(result[k])})")

    return result

def get_search_strategy(model_key, tuning_mode):
    """
    Decide hyperparameter search strategy
    """
    if tuning_mode == "manual":
        return None

    if model_key == "logistic_regression":
        return "random"

    if model_key == "lasso":
        return "random"

    if model_key in LINEAR_MODELS:
        return "grid"

    if model_key in NON_LINEAR_MODELS:
        return "random"

    return None

def validate_logistic_regression_before_training(pipeline, X, y):
    """
    Validate logistic regression parameters before training to prevent common errors
    """
    try:
        # Get the model from pipeline
        model = pipeline.named_steps.get('model')

        if model is None or not hasattr(model, 'solver'):
            return True  # Not a logistic regression model

        # Check for potential issues
        warnings = []

        # 1. Check for infinite values
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if np.any(np.isinf(X[col])):
                    warnings.append(f"Column '{col}' contains infinite values")

        # 2. Check for perfect separation warning
        if hasattr(model, 'solver') and model.solver == 'lbfgs':
            # lbfgs can have convergence issues with perfect separation
            pass

        # 3. Check class distribution
        if hasattr(y, 'value_counts'):
            class_counts = y.value_counts()
            if len(class_counts) < 2:
                warnings.append("Target has only one class")
            elif class_counts.min() / class_counts.sum() < 0.1:
                warnings.append("Severe class imbalance detected")

        if warnings:
            pass  # placeholder (debug print removed)
            # [DEBUG] print(f"  ⚠️ Logistic regression warnings: {warnings}")

        return True

    except Exception as e:
        # [DEBUG] print(f"  ❌ Logistic regression validation failed: {e}")
        return False

def split_columns(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    return numeric_cols, categorical_cols

def normalize_task_type(task_type):
    if "classification" in task_type:
        return "classification"
    return task_type

def normalize_null_params(params):
    """
    Convert string 'null' / 'None' / '' to Python None
    """
    for k, v in list(params.items()):
        if isinstance(v, str) and v.lower() in ("null", "none", ""):
            params[k] = None
    return params

def normalize_params(model_key, raw_params):
    normalized = []

    for name, cfg in raw_params.items():

        # CASE 1: already structured
        if isinstance(cfg, dict):
            param = cfg.copy()
            param["name"] = name

        # CASE 2: list → categorical
        elif isinstance(cfg, list):
            param = {
                "name": name,
                "type": "categorical",
                "values": cfg,
                "default": cfg[0] if len(cfg) > 0 else None
            }

        else:
            continue

        # Ensure type exists
        param.setdefault("type", "categorical")

        # Ensure default exists
        param.setdefault("default", None)

        # UI-safe ID
        param["ui_id"] = f"{model_key}__{name}"

        normalized.append(param)

    return normalized

def safe_r2_score(y_true, y_pred):
    """
    Safe R² calculation that handles NaN and edge cases
    """
    # Remove NaN and Inf from both arrays
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))

    if np.sum(mask) < 2:
        return np.nan  # Not enough valid samples

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    # Check for constant target
    if np.std(y_true_clean) < 1e-10:
        return np.nan  # Can't compute R² for constant target

    try:
        return r2_score(y_true_clean, y_pred_clean)
    except:
        return np.nan

def safe_numeric_conversion(series):
    """OPTIMIZED numeric conversion - Much faster"""
    if is_numeric_dtype(series):
        return series

    # For very large series, sample first to determine pattern
    if len(series) > 10000:
        sample = series.dropna().head(1000)
        if len(sample) > 0:
            # Check patterns in sample
            sample_str = sample.astype(str)
            has_percent = sample_str.str.contains('%', regex=False).any()
            has_currency = sample_str.str.contains(r'[$€£]', regex=True).any()
            has_commas = sample_str.str.contains(',', regex=False).any()

            if has_percent or has_currency or has_commas:
                # Apply cleaning to entire series
                cleaned = series.astype(str)
                if has_percent:
                    cleaned = cleaned.str.replace('%', '', regex=False)
                if has_currency:
                    cleaned = cleaned.str.replace(r'[$€£]', '', regex=True)
                if has_commas:
                    cleaned = cleaned.str.replace(',', '', regex=False)
                cleaned = cleaned.str.strip()
                return pd.to_numeric(cleaned, errors='coerce')

    # Fast path for most cases
    return pd.to_numeric(series, errors='coerce')

def convert_param_value(param_name, value):
    """
    Convert parameter value to appropriate type
    """
    if isinstance(value, str):
        # Handle null/None
        if value.lower() in ['null', 'none', '']:
            return None

        # Handle boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        # Handle numeric
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    return value

def update_training_progress(current, total, model_name):
    pct = int((current / total) * 100)
    set_training_status(
        current_model=model_name,
        model_index=current,
        total_models=total,
        message=f"Training {model_name}",
    )
    TRAINING_STATUS["progress"] = pct

def set_training_status(**kwargs):
    """Update training status with proper JSON serializable values"""
    global TRAINING_STATUS

    for key, value in kwargs.items():
        # Convert numpy types to Python types
        if isinstance(value, (np.integer, np.floating)):
            TRAINING_STATUS[key] = float(value)
        elif isinstance(value, np.ndarray):
            TRAINING_STATUS[key] = value.tolist()
        elif isinstance(value, pd.Timestamp):
            TRAINING_STATUS[key] = value.isoformat()
        else:
            TRAINING_STATUS[key] = value

    # Add timestamp
    TRAINING_STATUS["last_update"] = time.time()

    # Debug logging
    # [DEBUG] print(f"Training Status Updated: {kwargs}")

def detect_column_type(series):
    """OPTIMIZED column type detection - Sample based for speed"""
    # Handle empty series
    if len(series) == 0:
        return "empty"

    # Sample for large datasets
    sample_size = min(1000, len(series))
    if len(series) > sample_size:
        sample = series.sample(n=sample_size, random_state=42)
    else:
        sample = series

    non_null = sample.dropna()
    if len(non_null) == 0:
        return "empty"

    # Check unique values in sample
    unique_vals = non_null.nunique()

    # Fast checks
    if unique_vals == 2:
        # FIX: Check if values are actually 0 and 1
        unique_values = non_null.unique()
        if set(unique_values) == {0, 1} or set(unique_values) == {0.0, 1.0}:
            return "binary"
        else:
            return "categorical"

    # For large datasets, use faster checks
    if len(series) > 5000:
        # Check if it's numeric by trying conversion on sample
        numeric_sample = pd.to_numeric(non_null, errors='coerce')
        numeric_ratio = numeric_sample.notna().mean()

        if numeric_ratio > 0.8:
            if unique_vals <= 15:
                return "numeric_low_cardinality"
            elif unique_vals <= 50:
                return "numeric_discrete"
            else:
                return "numeric"
        else:
            if unique_vals <= 20:
                return "categorical"
            else:
                return "categorical_high_cardinality"

    # Original logic for smaller datasets
    if unique_vals <= 15:
        numeric = safe_numeric_conversion(non_null)
        # FIX: Use .all() for boolean check
        if numeric.notna().mean() > 0.9 and numeric.nunique() <= 15:
            if numeric.nunique() <= 5:
                return "ordinal"
            else:
                return "numeric_low_cardinality"
        return "categorical"

    numeric = safe_numeric_conversion(non_null)
    if numeric.notna().mean() > 0.9:
        if numeric.nunique() <= 50:
            return "numeric_discrete"
        else:
            skew_val = numeric.skew()
            if abs(skew_val) > 1:
                return "numeric_skewed"
            else:
                return "numeric"

    # Quick datetime check
    if len(non_null) > 0:
        try:
            pd.to_datetime(non_null.head(10), errors='raise')
            return "datetime"
        except:
            pass

    if unique_vals > 100:
        return "categorical_high_cardinality"

    return "categorical"

def auto_clean_dataset(df):
    """
    Fully automatic dataset cleaning before any processing
    Returns cleaned DataFrame and cleaning report
    """
    original_shape = df.shape
    df_clean = df.copy()
    cleaning_report = []

    try:
        # 1. Clean column names
        df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
        cleaning_report.append(f"Cleaned {len(df.columns)} column names")

        # 2. Remove completely empty columns
        empty_cols = df_clean.columns[df_clean.isna().all()].tolist()
        if empty_cols:
            df_clean.drop(columns=empty_cols, inplace=True)
            cleaning_report.append(f"Dropped {len(empty_cols)} completely empty columns: {', '.join(empty_cols[:5])}" +
                                 (f" and {len(empty_cols)-5} more" if len(empty_cols) > 5 else ""))

        # 3. Remove completely empty rows
        empty_rows = df_clean.isna().all(axis=1)
        if empty_rows.any():
            df_clean = df_clean[~empty_rows]
            cleaning_report.append(f"Removed {empty_rows.sum()} completely empty rows")

        # 4. Remove duplicate rows (only if reasonable size)
        if len(df_clean) < 100000:  # Only check duplicates for reasonable sized datasets
            dup_count = df_clean.duplicated().sum()
            if dup_count > 0:
                df_clean = df_clean.drop_duplicates()
                cleaning_report.append(f"Removed {dup_count} duplicate rows")

        # 5. Remove constant columns (all same value)
        constant_cols = []
        for col in df_clean.columns:
            if df_clean[col].nunique() == 1:
                constant_cols.append(col)
        if constant_cols:
            df_clean.drop(columns=constant_cols, inplace=True)
            cleaning_report.append(f"Dropped {len(constant_cols)} constant columns: {', '.join(constant_cols[:3])}" +
                                 (f" and {len(constant_cols)-3} more" if len(constant_cols) > 3 else ""))

        # 6. Convert string numbers to numeric (only first 1000 rows for speed)
        for col in df_clean.select_dtypes(include=['object']).columns:
            # Try to convert to numeric
            sample = df_clean[col].head(1000) if len(df_clean) > 1000 else df_clean[col]
            numeric = pd.to_numeric(sample, errors='coerce')
            if numeric.notna().mean() > 0.9:  # If >90% successful conversion
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                cleaning_report.append(f"Converted '{col}' from string to numeric")

        # 7. Summary statistics
        total_missing = df_clean.isna().sum().sum()
        total_cells = df_clean.shape[0] * df_clean.shape[1]
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        cleaning_report.append(f"Final dataset: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
        cleaning_report.append(f"Total missing values: {total_missing} ({missing_pct:.1f}%)")
        cleaning_report.append(f"Rows removed: {original_shape[0] - df_clean.shape[0]}")
        cleaning_report.append(f"Columns removed: {original_shape[1] - df_clean.shape[1]}")

    except Exception as e:
        cleaning_report.append(f"Error during cleaning: {str(e)}")
        # Return original if cleaning fails
        df_clean = df

    return df_clean, cleaning_report

def build_preprocessor(X):
    numeric_cols, categorical_cols = split_columns(X)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor

def check_feature_scaling_needed(X):
    """
    Check if features need scaling to avoid convergence issues
    Returns True if scaling is recommended
    """
    numeric_cols = X.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        return False

    # Check variance ratios
    variances = X[numeric_cols].var()
    max_var = variances.max()
    min_var = variances.min()

    # If variance ratio > 1000, scaling is needed
    if max_var > 0 and min_var > 0:
        variance_ratio = max_var / min_var
        return variance_ratio > 1000

    return False

def build_pipeline(model_cls, X):
    """Enhanced pipeline builder with better handling for large datasets"""
    numeric_cols, categorical_cols = split_columns(X)


    # For very large datasets, use simpler imputation
    if len(X) > 100000:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", RobustScaler())  # More robust for large datasets
        ])

    # For categorical columns with high cardinality, use frequency encoding
    if categorical_cols:
        # Check for high cardinality columns
        high_card_cols = []
        low_card_cols = []

        for col in categorical_cols:
            if X[col].nunique() > 50:
                high_card_cols.append(col)
            else:
                low_card_cols.append(col)

        # Different pipelines for different cardinalities
        if low_card_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    max_categories=50  # Limit categories
                ))
            ])
        else:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ))
            ])

        transformers = [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, low_card_cols)
        ]
    else:
        transformers = [
            ("num", numeric_pipeline, numeric_cols)
        ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=-1 if len(X) > 10000 else None  # Parallel processing for large datasets
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_cls())
    ])

def optimize_target(y, task_type):
    info = {
        "applied": False,
        "method": None,
        "skewness": None
    }

    if task_type != "regression":
        return y, info

    skew = y.skew()
    info["skewness"] = float(skew)

    if skew > 1 and (y > 0).all():
        y = np.log1p(y)
        info["applied"] = True
        info["method"] = "log1p"

    return y, info

def generate_eda_report(df):
    """Comprehensive EDA report with JSON serializable types"""
    report = {
        "overview": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "memory_usage": float(df.memory_usage(deep=True).sum() / 1024 / 1024),  # MB
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing": int(df.isnull().sum().sum()),
            "missing_percentage": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        },
        "columns": {}
    }

    for col in df.columns:
        col_data = df[col]
        col_type = detect_column_type(col_data)

        column_info = {
            "type": col_type,
            "non_null": int(col_data.notna().sum()),
            "null_count": int(col_data.isna().sum()),
            "null_percentage": float(col_data.isna().sum() / len(col_data) * 100),
            "unique_values": int(col_data.nunique()),
            "sample_values": [str(val) for val in col_data.dropna().head(5).tolist()] if col_data.nunique() > 1 else [str(col_data.iloc[0])]
        }

        # Type-specific statistics (convert numpy types to Python types)
        if col_type in ["numeric", "numeric_skewed", "numeric_discrete", "numeric_low_cardinality"]:
            numeric = safe_numeric_conversion(col_data.dropna())
            if len(numeric) > 0:
                column_info.update({
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "std": float(numeric.std()),
                    "skewness": float(numeric.skew()),
                    "kurtosis": float(numeric.kurtosis()),
                    "q1": float(numeric.quantile(0.25)),
                    "q3": float(numeric.quantile(0.75)),
                    "iqr": float(numeric.quantile(0.75) - numeric.quantile(0.25))
                })

        elif col_type in ["categorical", "binary", "ordinal"]:
            value_counts = col_data.value_counts().head(10)
            # Convert to regular Python dict with string keys
            column_info["top_values"] = {str(k): int(v) for k, v in value_counts.to_dict().items()}
            column_info["value_counts_total"] = int(len(value_counts))

        report["columns"][str(col)] = column_info

    return report

def create_visualizations(df, target_col=None):
    """Generate comprehensive visualizations"""
    viz_data = {}

    # 1. Missing values heatmap
    fig_missing = plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig_missing)
    buf.seek(0)
    viz_data["missing_heatmap"] = base64.b64encode(buf.read()).decode('utf-8')

    # 2. Distribution plots for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        n_cols = min(4, len(numeric_cols))
        n_rows = math.ceil(len(numeric_cols) / n_cols)

        fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], bins=30)
                axes[idx].set_title(f'{col} Distribution')
                axes[idx].tick_params(axis='x', rotation=45)

        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig_dist)
        buf.seek(0)
        viz_data["distributions"] = base64.b64encode(buf.read()).decode('utf-8')

    # 3. Correlation heatmap if multiple numeric columns
    if len(numeric_cols) > 1:
        fig_corr = plt.figure(figsize=(12, 10))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig_corr)
        buf.seek(0)
        viz_data["correlation"] = base64.b64encode(buf.read()).decode('utf-8')

    # 4. Box plots for outliers
    if numeric_cols:
        fig_box = plt.figure(figsize=(15, 6))
        df_numeric = df[numeric_cols].select_dtypes(include=[np.number])
        df_melted = pd.melt(df_numeric)
        sns.boxplot(x='variable', y='value', data=df_melted)
        plt.title("Box Plots (Check for Outliers)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig_box)
        buf.seek(0)
        viz_data["boxplots"] = base64.b64encode(buf.read()).decode('utf-8')

    # 5. Target vs features if target provided
    if target_col and target_col in df.columns:
        if is_numeric_dtype(df[target_col]):
            # Scatter plots with top correlated features
            corr_with_target = df.corr()[target_col].abs().sort_values(ascending=False)
            top_features = corr_with_target.index[1:4]  # Top 3 excluding target itself

            if len(top_features) > 0:
                fig_target, axes = plt.subplots(1, len(top_features), figsize=(15, 4))
                if len(top_features) == 1:
                    axes = [axes]

                for idx, feature in enumerate(top_features):
                    if feature in df.columns:
                        axes[idx].scatter(df[feature], df[target_col], alpha=0.5)
                        axes[idx].set_xlabel(feature)
                        axes[idx].set_ylabel(target_col)
                        axes[idx].set_title(f'{feature} vs {target_col}')

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig_target)
                buf.seek(0)
                viz_data["target_relationships"] = base64.b64encode(buf.read()).decode('utf-8')

    return viz_data

def determine_task_type(series, index=None):
    """
    FINAL task detection
    """

    meta = {
        "zero_ratio": 0.0,
        "unique_values": int(series.nunique(dropna=True))
    }

    if is_numeric_dtype(series):
        zero_ratio = float((series == 0).mean())
        meta["zero_ratio"] = zero_ratio

        if 0.3 < zero_ratio < 0.95:
            non_zero_unique = series[series != 0].nunique()
            if non_zero_unique > 10:
                return "zero_inflated_regression", series, meta
            else:
                return "zero_inflated_classification", series, meta

        unique_vals = set(series.dropna().unique())
        if unique_vals in [{0, 1}, {0.0, 1.0}]:
            return "binary_classification", series, meta

        if series.nunique() <= 10:
            return "multiclass_classification", series, meta

        return "regression", series, meta

    if is_string_dtype(series):
        return (
            "binary_classification" if series.nunique() == 2
            else "multiclass_classification",
            series,
            meta
        )

    return "regression", series, meta

def prepare_target(series, index=None):
    task_type, cleaned_series, meta = determine_task_type(series)

    target_info = {
        "type": task_type,
        "processed_target": None,
        "binary_target": None,
        "regression_target": None,
        "regression_target_transformed": None,
        "classification_target": None,
        "zero_ratio": meta["zero_ratio"],
        "unique_values": meta["unique_values"]
    }

    y = cleaned_series.dropna()

    if task_type == "zero_inflated_regression":
        # Create binary target
        target_info["binary_target"] = (y > 0).astype(int)
        non_zero = y[y > 0]

        # FIX: Add non_zero_samples count
        target_info["non_zero_samples"] = len(non_zero)

        if len(non_zero) > 0:
            # Calculate statistics
            if non_zero.skew() > 1 and (non_zero > 0).all():
                target_info["regression_target"] = non_zero
                target_info["regression_target_transformed"] = np.log1p(non_zero)
                target_info["transformation"] = "log1p"

                # Add statistics
                target_info["non_zero_stats"] = {
                    "mean": float(non_zero.mean()),
                    "std": float(non_zero.std()),
                    "skew": float(non_zero.skew()),
                    "min": float(non_zero.min()),
                    "max": float(non_zero.max())
                }
            else:
                target_info["regression_target"] = non_zero
                target_info["transformation"] = "none"
        else:
            target_info["non_zero_samples"] = 0
            target_info["transformation"] = "none"

    elif task_type == "zero_inflated_classification":
        target_info["binary_target"] = (y > 0).astype(int)
        target_info["classification_target"] = y
        # FIX: Add non_zero_samples count
        target_info["non_zero_samples"] = len(y[y > 0])

    elif "classification" in task_type:
        le = LabelEncoder()
        encoded = le.fit_transform(y.astype(str))
        target_info["processed_target"] = pd.Series(encoded, index=y.index)
        target_info["class_mapping"] = dict(zip(le.classes_, le.transform(le.classes_)))

    else:
        target_info["processed_target"] = y

    return target_info

def validate_and_fix_hyperparameters(model_key, manual_params, param_cfg):
    """
    Validate and fix hyperparameters for manual tuning mode
    Returns validated parameters dictionary
    """
    # [DEBUG] print(f"  -> Validating hyperparameters for {model_key}")
    # [DEBUG] print(f"  -> Manual params received: {manual_params}")

    validated_params = {}

    # First, clean all parameters by converting string values to appropriate types
    for key, value in manual_params.items():
        # Handle comma-separated values (take first value)
        original_value = value
        if isinstance(value, str) and ',' in value:
            # SPECIAL CASE: If this is a categorical parameter with multiple options, keep as is
            is_categorical = any(param in key for param in ['max_features', 'solver', 'loss', 'criterion',
                                                            'penalty', 'selection', 'algorithm', 'weights',
                                                            'metric'])

            if not is_categorical:
                # Take first non-empty value for non-categorical parameters
                parts = [part.strip() for part in value.split(',') if part.strip()]
                if parts:
                    value = parts[0]
                    # [DEBUG] print(f"    -> Fixed comma-separated '{original_value}' -> '{value}' for {key}")
                else:
                    value = ''

        if isinstance(value, str):
            # Handle null/None values
            if value.lower() in ['null', 'none', '', 'undefined', 'nan']:
                # For certain parameters, None is valid
                if key.endswith(('max_depth', 'max_samples', 'max_features')):
                    validated_params[key] = None
                else:
                    validated_params[key] = value
                continue

            # Handle boolean values - CRITICAL FIX
            bool_params = ['copy_X', 'fit_intercept', 'positive', 'bootstrap', 'class_weight']
            is_bool_param = any(param in key for param in bool_params)

            if is_bool_param:
                # Special handling for class_weight (can be string or None)
                if 'class_weight' in key:
                    if value.lower() in ['null', 'none', '']:
                        validated_params[key] = None
                    elif value in ['balanced', 'balanced_subsample']:
                        validated_params[key] = value
                    else:
                        validated_params[key] = None  # Default to None
                else:
                    # Regular boolean parameter
                    if value.lower() in ['true', '1', 'yes', 'on', 't']:
                        validated_params[key] = True
                    elif value.lower() in ['false', '0', 'no', 'off', 'f']:
                        validated_params[key] = False
                    else:
                        # Try to infer from common patterns
                        if 'true' in value.lower():
                            validated_params[key] = True
                        elif 'false' in value.lower():
                            validated_params[key] = False
                        else:
                            validated_params[key] = True  # Default to True
                            # [DEBUG] print(f"    -> WARNING: Could not parse boolean '{value}' for {key}, defaulting to True")
                continue

            # Handle numeric values
            try:
                if '.' in value or 'e' in value.lower():
                    # Float value
                    validated_params[key] = float(value)
                else:
                    # Integer value
                    validated_params[key] = int(value)
                continue
            except (ValueError, TypeError):
                # Not a number, check if it's a special string value
                if value in ['sqrt', 'log2', 'auto', 'scale']:
                    validated_params[key] = value
                else:
                    # Keep as string for categorical parameters
                    validated_params[key] = value
        else:
            # Non-string value, keep as is
            validated_params[key] = value

    # [DEBUG] print(f"  -> Cleaned params: {validated_params}")

       # Model-specific validation and fixing
    if model_key == "logistic_regression":
        # Use the dedicated fix_logistic_params function
        validated_params = fix_logistic_params(validated_params)

    elif model_key == "linear_regression":
        # SPECIAL FIX FOR LINEAR REGRESSION BOOLEAN PARAMETERS
        for param in ["model__copy_X", "model__fit_intercept", "model__positive"]:
            if param in validated_params:
                value = validated_params[param]
                if isinstance(value, bool):
                    continue  # Already boolean
                elif isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes', 'on', 't']:
                        validated_params[param] = True
                    elif value.lower() in ['false', '0', 'no', 'off', 'f']:
                        validated_params[param] = False
                    else:
                        validated_params[param] = True  # Default
                else:
                    validated_params[param] = bool(value)  # Convert to boolean

        # [DEBUG] print(f"  -> Fixed linear regression params: {validated_params}")

    elif model_key == "lasso":
        # Ensure max_iter is high enough for convergence
        if "model__max_iter" in validated_params:
            max_iter = validated_params["model__max_iter"]
            if isinstance(max_iter, (int, float)) and max_iter < 5000:
                validated_params["model__max_iter"] = 5000
                # [DEBUG] print(f"    -> Increased max_iter to 5000 for Lasso convergence")

        # Ensure tol is not too tight
        if "model__tol" in validated_params:
            tol = validated_params["model__tol"]
            if isinstance(tol, (int, float)) and tol < 1e-4:
                validated_params["model__tol"] = 1e-4
                # [DEBUG] print(f"    -> Increased tol to 1e-4 for Lasso convergence")

    elif model_key == "ridge":
        # Fix boolean parameters
        for param in ["model__copy_X", "model__fit_intercept"]:
            if param in validated_params:
                value = validated_params[param]
                if isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes', 'on', 't']:
                        validated_params[param] = True
                    elif value.lower() in ['false', '0', 'no', 'off', 'f']:
                        validated_params[param] = False
                    else:
                        validated_params[param] = True  # Default

        # Ensure alpha is positive float
        alpha = validated_params.get("model__alpha", 1.0)
        try:
            alpha = float(alpha)
            if alpha <= 0:
                alpha = 1.0
                # [DEBUG] print(f"    -> Fixed alpha from {validated_params.get('model__alpha')} to 1.0 (must be positive)")
        except:
            alpha = 1.0
            # [DEBUG] print(f"    -> Fixed invalid alpha to 1.0")

        validated_params["model__alpha"] = alpha

        # Ensure max_iter is int
        if "model__max_iter" in validated_params:
            try:
                validated_params["model__max_iter"] = int(float(validated_params["model__max_iter"]))
            except:
                validated_params["model__max_iter"] = 1000

    elif model_key in ["random_forest_regressor", "random_forest_classifier"]:
        # Fix bootstrap parameter - CRITICAL FIX
        if "model__bootstrap" in validated_params:
            value = validated_params["model__bootstrap"]
            if isinstance(value, str):
                # Handle comma-separated
                if ',' in value:
                    value = value.split(',')[0].strip()

                if value.lower() in ['true', '1', 'yes', 'on', 't']:
                    validated_params["model__bootstrap"] = True
                elif value.lower() in ['false', '0', 'no', 'off', 'f']:
                    validated_params["model__bootstrap"] = False
                else:
                    validated_params["model__bootstrap"] = True  # Default
                    # [DEBUG] print(f"    -> Fixed bootstrap to True (default)")
            elif not isinstance(value, bool):
                validated_params["model__bootstrap"] = True  # Default

        # Fix n_estimators
        if "model__n_estimators" in validated_params:
            try:
                n_estimators = int(float(validated_params["model__n_estimators"]))
                if n_estimators < 1:
                    n_estimators = 100
                    # [DEBUG] print(f"    -> Fixed n_estimators to 100 (must be >= 1)")
                validated_params["model__n_estimators"] = n_estimators
            except:
                validated_params["model__n_estimators"] = 100
                # [DEBUG] print(f"    -> Fixed invalid n_estimators to 100")

        # Handle max_depth
        if "model__max_depth" in validated_params:
            md_value = validated_params["model__max_depth"]
            if md_value is None or (isinstance(md_value, str) and md_value.lower() in ['null', 'none', '']):
                validated_params["model__max_depth"] = None
            else:
                try:
                    validated_params["model__max_depth"] = int(float(md_value))
                except:
                    validated_params["model__max_depth"] = None
                    # [DEBUG] print(f"    -> Fixed max_depth to None")

        # Handle max_features
        if "model__max_features" in validated_params:
            mf_value = validated_params["model__max_features"]
            if mf_value is None or (isinstance(mf_value, str) and mf_value.lower() in ['null', 'none', '']):
                validated_params["model__max_features"] = None
            elif isinstance(mf_value, str):
                # Clean comma-separated
                if ',' in mf_value:
                    parts = [part.strip() for part in mf_value.split(',') if part.strip()]
                    if parts:
                        validated_params["model__max_features"] = parts[0]

        # Ensure min_samples_split, min_samples_leaf are int
        for param in ["model__min_samples_split", "model__min_samples_leaf"]:
            if param in validated_params:
                try:
                    validated_params[param] = int(float(validated_params[param]))
                except:
                    validated_params[param] = 2 if param.endswith("split") else 1

    elif model_key in ["gradient_boosting_regressor", "gradient_boosting_classifier"]:
        # Fix criterion parameter - CRITICAL FIX
        if "model__criterion" in validated_params:
            criterion = validated_params["model__criterion"]
            if isinstance(criterion, str):
                # Clean comma-separated values
                if ',' in criterion:
                    parts = [part.strip() for part in criterion.split(',') if part.strip()]
                    if parts:
                        criterion = parts[0]

                # Validate against allowed values
                if model_key == "gradient_boosting_regressor":
                    valid_criteria = ['friedman_mse', 'squared_error']
                    default_criterion = 'friedman_mse'
                else:
                    valid_criteria = ['friedman_mse', 'squared_error']
                    default_criterion = 'friedman_mse'

                if criterion not in valid_criteria:
                    validated_params["model__criterion"] = default_criterion
                    # [DEBUG] print(f"    -> Fixed invalid criterion '{criterion}' to '{default_criterion}'")

        # Fix loss parameter for regressor
        if model_key == "gradient_boosting_regressor" and "model__loss" in validated_params:
            loss_value = validated_params["model__loss"]
            if isinstance(loss_value, str) and ',' in loss_value:
                parts = [part.strip() for part in loss_value.split(',') if part.strip()]
                if parts:
                    validated_params["model__loss"] = parts[0]

    elif model_key in ["knn_regressor", "knn_classifier"]:
        # Ensure n_neighbors is int
        if "model__n_neighbors" in validated_params:
            try:
                validated_params["model__n_neighbors"] = int(float(validated_params["model__n_neighbors"]))
                if validated_params["model__n_neighbors"] < 1:
                    validated_params["model__n_neighbors"] = 5
            except:
                validated_params["model__n_neighbors"] = 5

        # Ensure leaf_size is int
        if "model__leaf_size" in validated_params:
            try:
                validated_params["model__leaf_size"] = int(float(validated_params["model__leaf_size"]))
                if validated_params["model__leaf_size"] < 1:
                    validated_params["model__leaf_size"] = 30
            except:
                validated_params["model__leaf_size"] = 30

        # Handle algorithm parameter
        if "model__algorithm" in validated_params:
            algo = validated_params["model__algorithm"]
            if isinstance(algo, str) and ',' in algo:
                parts = [part.strip() for part in algo.split(',') if part.strip()]
                if parts:
                    validated_params["model__algorithm"] = parts[0]

    # [DEBUG] print(f"  -> Validated params: {validated_params}")
    return validated_params

def run_training(X, y, task_type, selected_features, form_dict):
    """
    Main training function called from the Flask route
    Handles both regular and zero-inflated training
    """
    global TRAINING_STATUS

    # [DEBUG] print(f"\n{'='*80}")
    # [DEBUG] print(f"RUN_TRAINING: Starting")
    # [DEBUG] print(f"Task type: {task_type}")
    # [DEBUG] print(f"X shape: {X.shape}")
    # [DEBUG] print(f"Form keys: {list(form_dict.keys())}")
    # [DEBUG] print(f"Selected features: {selected_features}")

    try:
        # Reset training status
        TRAINING_STATUS.update({
            "running": True,
            "done": False,
            "current_model": None,
            "model_index": 0,
            "total_models": 0,
            "message": "Starting training...",
            "started_at": time.time()
        })

        # Check if it's zero-inflated task
        if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
            # [DEBUG] print(f"Running zero-inflated training...")

            # Ensure we have target_info
            if isinstance(y, dict):
                target_info = y
            else:
                # Try to get from DATASTORE
                target_info = DATASTORE.get("target_info")
                if not target_info:
                    # Try to reconstruct
                    y_binary = DATASTORE.get("y_binary")
                    if y_binary is not None:
                        target_info = {
                            "type": task_type,
                            "binary_target": y_binary,
                            "zero_ratio": (y_binary == 0).mean() if hasattr(y_binary, 'mean') else 0.5,
                            "non_zero_samples": len(y_binary[y_binary == 1]) if hasattr(y_binary, 'sum') else 0
                        }
                    else:
                        raise ValueError("Cannot run zero-inflated training: missing target info")

            # [DEBUG] print(f"Target info: type={target_info.get('type')}, zero_ratio={target_info.get('zero_ratio')}, non_zero_samples={target_info.get('non_zero_samples')}")

            # Run two-stage training
            results = train_zero_inflated_models(
                X, target_info, selected_features, form_dict
            )

        else:
            # [DEBUG] print(f"Running regular training...")

            # Get appropriate y
            y_data = None
            if isinstance(y, pd.Series):
                y_data = y
            elif isinstance(y, dict) and "processed_target" in y:
                y_data = y["processed_target"]
            else:
                # Try to get from DATASTORE
                y_data = DATASTORE.get("y_aligned") or DATASTORE.get("y")

            if y_data is None:
                raise ValueError(f"No target data available for training. y type: {type(y)}")

            # [DEBUG] print(f"y_data type: {type(y_data)}, length: {len(y_data)}")

            # Run regular training
            results = train_models_with_manual_control(
                X, y_data, task_type, selected_features, form_dict
            )

        # Store results
        DATASTORE["training_results"] = results

        # Update training status
        TRAINING_STATUS.update({
            "running": False,
            "done": True,
            "message": f"Training completed successfully"
        })

        # [DEBUG] print(f"\nTraining completed successfully!")
        # [DEBUG] print(f"Results type: {type(results)}")

    except Exception as e:
        error_msg = str(e)
        # [DEBUG] print(f"\n❌ ERROR in run_training: {error_msg}")
        import traceback
        traceback.print_exc()

        TRAINING_STATUS.update({
            "running": False,
            "done": True,
            "message": f"Training failed: {error_msg[:100]}",
            "error": error_msg
        })

        # Store error in results
        DATASTORE["training_results"] = {
            "error": error_msg,
            "type": "error"
        }

def optimized_feature_engineering(X, y=None, task_type="regression"):
    """
    Optimized feature engineering for NON time-series data
    Works for regression, classification, binary, multiclass, zero-inflated
    """

    X = X.copy() # prevents modification of original_data

    fe_info = {
        "rows_before": X.shape[0],
        "rows_dropped": 0,
        "features_before": X.shape[1],
        "features_after": None,

        "numeric_imputation_cols": [],
        "categorical_imputation_cols": [],
        "one_hot_encoded_cols": [],
        "frequency_encoded_cols": [],

        "notes": []
    }

    # ----------------------------
    # Identify column types
    # ----------------------------
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # ----------------------------
    # Numeric Imputation
    # ----------------------------
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())   # fill na values with median
            fe_info["numeric_imputation_cols"].append(col)

    # ----------------------------
    # Categorical Imputation
    # ----------------------------
    for col in categorical_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode().iloc[0])    # fill categorical na values with mode()
            fe_info["categorical_imputation_cols"].append(col)

    # ----------------------------
    # Encoding Strategy
    # ----------------------------
    for col in categorical_cols:
        unique_vals = X[col].nunique()

        # Low cardinality → One Hot (Cardinality -> unique values)
        if unique_vals <= 10:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            fe_info["one_hot_encoded_cols"].append(col)

        # Medium cardinality → Frequency Encoding
        elif 10 < unique_vals <= 50:
            freq_map = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq_map)
            fe_info["frequency_encoded_cols"].append(col)

        # High cardinality → Drop
        else:
            X.drop(columns=[col], inplace=True)
            fe_info["notes"].append(f"Dropped high-cardinality column: {col}")

    fe_info["features_after"] = X.shape[1]

    if not fe_info["numeric_imputation_cols"]:
        fe_info["notes"].append("No numeric imputation required")
    if not fe_info["categorical_imputation_cols"]:
        fe_info["notes"].append("No categorical imputation required")

    return X, fe_info

def advanced_feature_engineering(X, y, task_type):
    """
    Advanced feature engineering for AUTO mode
    Simplified version for speed
    """
    # [DEBUG] print(f"Running advanced feature engineering...")

    X_clean = X.copy()

    # Remove any pandas index columns
    cols_to_remove = []
    for col in X_clean.columns:
        col_lower = str(col).lower()
        if (col_lower.startswith('unnamed') or
            col_lower == 'index' or
            col_lower == 'id'):
            cols_to_remove.append(col)

    if cols_to_remove:
        X_clean = X_clean.drop(columns=cols_to_remove)

    # Impute missing values
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)

    # Encode categorical columns
    cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if X_clean[col].nunique() <= 10:
            # One-hot encode
            dummies = pd.get_dummies(X_clean[col], prefix=col, drop_first=True)
            X_clean = pd.concat([X_clean.drop(columns=[col]), dummies], axis=1)
        else:
            # Drop high cardinality
            X_clean = X_clean.drop(columns=[col])

    # Align with y
    common_idx = X_clean.index.intersection(y.index)
    X_clean = X_clean.loc[common_idx]
    y_aligned = y.loc[common_idx]

    transformations = {
        "removed_columns": cols_to_remove,
        "imputed_numeric": len(numeric_cols),
        "encoded_categorical": len(cat_cols),
        "final_shape": X_clean.shape
    }

    feature_metadata = {
        "processing_steps": [
            {"name": "remove_index_columns", "status": "completed"},
            {"name": "impute_numeric", "status": "completed"},
            {"name": "encode_categorical", "status": "completed"},
            {"name": "align_data", "status": "completed"}
        ]
    }

    return X_clean, transformations, feature_metadata

def run_feature_engineering_async(X, y, task_type):
    global PROCESS_STATUS
    import state
    try:
        state.PROCESS_STATUS.update({
            "progress": 10,
            "stage": "running",
            "operation": "Feature Engineering",
            "message": "Processing features",
            "done": False,
            "error": None
        })

        # [DEBUG] print(f"DEBUG: Starting feature engineering for task: {task_type}")
        # [DEBUG] print(f"DEBUG: X shape: {X.shape}, y length: {len(y)}")

        # Call the simplified unified feature engineering
        X_processed, fe_results = unified_feature_engineering(X, y, task_type)

        # [DEBUG] print(f"DEBUG: Feature engineering completed")
        # [DEBUG] print(f"DEBUG: X_processed shape: {X_processed.shape}")

        # Align with target
        common_idx = X_processed.index.intersection(y.index)
        X_processed = X_processed.loc[common_idx]
        y_aligned = y.loc[common_idx]

        # [DEBUG] print(f"DEBUG: After alignment - X: {X_processed.shape}, y: {y_aligned.shape}")

        DATASTORE.update({
            "X_processed": X_processed,
            "y_aligned": y_aligned,
            "feature_engineering_summary": fe_results
        })

        state.PROCESS_STATUS.update({
            "progress": 100,
            "stage": "complete",
            "operation": "Completed",
            "message": "Feature engineering completed",
            "done": True,
            "error": None
        })
        # [DEBUG] print(f"DEBUG: Feature engineering stored in DATASTORE")

    except Exception as e:
        # [DEBUG] print(f"ERROR in feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()

        state.PROCESS_STATUS.update({
            "progress": 100,
            "stage": "error",
            "operation": "Failed",
            "message": str(e),
            "done": True,
            "error": str(e)
        })

def intelligent_feature_selection(X, y, task_type, n_features='auto', fast_mode=True):
    X_num = X.select_dtypes(include=np.number)

    if X_num.empty:
        raise ValueError("No numeric features available")

    if n_features == 'auto':
        n_features = min(20, X_num.shape[1])

    scores = {}
    for col in X_num.columns:
        try:
            corr = X_num[col].corr(y)
            if not np.isnan(corr):
                scores[col] = abs(corr)
        except:
            scores[col] = 0

    scores = pd.Series(scores).sort_values(ascending=False)
    selected = scores.head(n_features).index.tolist()

    return selected, scores.to_dict(), "Correlation-based", None

def create_feature_importance_plot(feature_scores, top_n=15):
    """
    Create feature importance visualization
    """
    top_features = list(feature_scores.keys())[:top_n]
    top_scores = list(feature_scores.values())[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores, color='steelblue')
    ax.set_yticks(y_pos)

    # Truncate long feature names
    labels = []
    for f in top_features:
        if len(f) > 30:
            labels.append(f[:27] + "...")
        else:
            labels.append(f)

    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {len(top_features)} Feature Importance Scores")
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")

def fast_feature_selection(X_numeric, y, task_type, n_features):
    """Ultra-fast feature selection for large datasets"""
    # [DEBUG] print("  -> Using ultra-fast correlation-based selection")

    if len(X_numeric.columns) <= n_features:
        return X_numeric.columns.tolist(), {}, "All features selected (fewer than requested)", None

    # Simple correlation-based selection
    scores = {}
    for col in X_numeric.columns:
        try:
            if task_type == "regression":
                corr = X_numeric[col].corr(y)
                if not np.isnan(corr):
                    scores[col] = abs(corr)
            else:
                # For classification, use correlation with encoded target
                if y.nunique() == 2:
                    # Binary: point-biserial correlation
                    y_numeric = y.astype(int)
                    corr = X_numeric[col].corr(y_numeric)
                    if not np.isnan(corr):
                        scores[col] = abs(corr)
                else:
                    # Multiclass: average absolute correlation per class
                    corr_sum = 0
                    for class_val in y.unique():
                        y_binary = (y == class_val).astype(int)
                        corr = X_numeric[col].corr(y_binary)
                        if not np.isnan(corr):
                            corr_sum += abs(corr)
                    scores[col] = corr_sum / y.nunique()
        except:
            scores[col] = 0

    scores_series = pd.Series(scores)
    scores_series = scores_series.sort_values(ascending=False)

    selected = scores_series.head(n_features).index.tolist()
    feature_scores = scores_series.to_dict()

    return selected, feature_scores, f"Fast Correlation-based (top {n_features})", None

def export_pipeline(pipeline, filename="trained_model.pkl"):
    joblib.dump(pipeline, filename)

def get_training_data(X, y, model_key, task_type, max_rows=25000):
    n = len(X)

    # Dynamic row limits based on model cost
    actual_max_rows = max_rows
    try:
        if isinstance(model_key, str):
            cost = MODEL_COST.get(model_key, 'medium')
            if cost == 'expensive':
                # SVR, RF, SVM are O(n^2) or worse, cap severely
                actual_max_rows = min(max_rows, 3000)
            elif cost == 'medium':
                actual_max_rows = min(max_rows, 10000)
    except:
        pass

    if n <= actual_max_rows:
        return X, y, False

    if task_type in ["binary_classification", "multiclass_classification"]:
        frac = actual_max_rows / n
        Xs, _, ys, _ = train_test_split(
            X, y,
            train_size=frac,
            stratify=y,
            random_state=42
        )
        return Xs, ys, True

    step = max(1, n // actual_max_rows)
    idx = X.iloc[::step].index
    return X.loc[idx], y.loc[idx], True

def cleanup_results_for_template(results):
    """
    Clean up results to prevent template errors - FIXED VERSION
    """
    if not isinstance(results, dict):
        return results

    # Ensure test_metrics exists
    if "test_metrics" not in results:
        results["test_metrics"] = {}

    # Ensure each model has proper structure
    for model_key, model_data in results.get("test_metrics", {}).items():
        if not isinstance(model_data, dict):
            # If it's not a dict, make it one
            results["test_metrics"][model_key] = {
                "label": str(model_key),
                "metrics": {}
            }
        else:
            # Ensure metrics exists and is a dict
            if "metrics" not in model_data:
                model_data["metrics"] = {}

            # Ensure label exists
            if "label" not in model_data:
                model_data["label"] = str(model_key)

            # Convert numpy types to Python types for JSON serialization
            if isinstance(model_data.get("metrics"), dict):
                for metric_key, metric_value in model_data["metrics"].items():
                    if isinstance(metric_value, (np.float64, np.float32, np.int64, np.int32)):
                        model_data["metrics"][metric_key] = float(metric_value)

    # Ensure best_model has proper structure
    if "best_model" in results and results["best_model"]:
        if not isinstance(results["best_model"], dict):
            results["best_model"] = {"label": "Best Model", "metrics": {}}
        elif "metrics" not in results["best_model"]:
            results["best_model"]["metrics"] = {}

    return results

def get_cv_split(X):

    length = len(X)

    if length > 100_000:
        return 2
    elif length>=50_000:
        return 3
    else:
        return 5

def run_hyperparameter_search(
    model_key, pipeline, X, y, param_cfg, scoring
):
    search_type = get_search_strategy(model_key, tuning_mode="auto")

    if not param_cfg or not search_type:
        pipeline.fit(X, y)
        return pipeline, None, None

    param_grid = build_param_grid(param_cfg)
    # Fast models can use standard CV, but expensive ones need fewer folds
    cv = 3 if model_key in ["svr", "random_forest_regressor", "random_forest_classifier", "gradient_boosting_regressor", "gradient_boosting_classifier", "svm"] else get_cv_split(X)

    if search_type == "grid":
        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1 # Use all available cores
        )

    else:  # randomized
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=5, # Reduced from 10 to 5 for speed
            scoring=scoring,
            cv=cv,
            random_state=42,
            n_jobs=-1 # Use all available cores
        )

    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_

def extract_manual_params(form_dict, model_key, param_cfg):
    """
    Extract manual parameters from form data - FIXED FOR LIST HANDLING
    """
    params = {}

    # Convert form data to simple dict
    form_dict_simple = {}
    for key, value in form_dict.items():
        if isinstance(value, list):
            # If it's a list with one element, extract it
            if len(value) == 1:
                form_dict_simple[key] = value[0]
            # If it's a list with multiple elements, check if they're all the same
            elif len(value) > 1 and all(v == value[0] for v in value):
                form_dict_simple[key] = value[0]  # Take first if all are same
            else:
                form_dict_simple[key] = value  # Keep as list
        else:
            form_dict_simple[key] = value

    # [DEBUG] print(f"  -> Looking for {model_key} parameters in form data...")

    # Check for each parameter in the config
    for param_name, cfg in param_cfg.items():

        # Try multiple key formats
        possible_keys = [
            f"{model_key}__{param_name}",  # logistic_regression__C
            f"{param_name}",  # just C
            param_name,  # model__C
            f"{model_key}_{param_name}",  # logistic_regression_C
        ]

        value = None
        for key in possible_keys:
            if key in form_dict_simple:
                value = form_dict_simple[key]
                # [DEBUG] print(f"    Found {param_name} as '{key}': {value}")
                break

        if value is not None and str(value).strip() != '':
            # Handle list values by taking first element
            if isinstance(value, list) and len(value) > 0:
                # Check if all values are the same
                if all(str(v) == str(value[0]) for v in value):
                    params[param_name] = value[0]
                    # [DEBUG] print(f"    -> Using first element from list: {value[0]}")
                else:
                    # Take first element as default
                    params[param_name] = value[0]
                    # [DEBUG] print(f"    -> Warning: List has different values, using first: {value[0]}")
            else:
                params[param_name] = value

    # Set defaults for models if nothing found
    if not params:
        # [DEBUG] print(f"  -> No parameters found for {model_key}, setting defaults")

        if model_key == "logistic_regression":
            params = {
                "model__solver": "lbfgs",
                "model__penalty": "l2",
                "model__C": 1.0,
                "model__max_iter": 1000
            }
        elif model_key == "random_forest_classifier":
            params = {
                "model__n_estimators": 100,
                "model__max_depth": None,
                "model__min_samples_split": 2,
                "model__bootstrap": True
            }
        elif model_key == "knn_classifier":
            params = {
                "model__n_neighbors": 5,
                "model__weights": "uniform",
                "model__p": 2
            }
        elif model_key == "lasso":
            params = {
                "model__alpha": 1.0,
                "model__fit_intercept": True,
                "model__max_iter": 1000
            }
        elif model_key == "random_forest_regressor":
            params = {
                "model__n_estimators": 100,
                "model__max_depth": None,
                "model__bootstrap": True
            }
        elif model_key == "adaboost_regressor":
            params = {
                "model__n_estimators": 50,
                "model__learning_rate": 1.0,
                "model__loss": "linear"
            }

    # [DEBUG] print(f"  -> Extracted {len(params)} parameters for {model_key}")
    return params

def get_model_specific_params(model_key, form_dict):
    """
    Get model-specific parameters from form data with enhanced handling
    """
    params = {}

    # Map model keys to parameter prefixes
    model_prefixes = {
        'linear_regression': 'linear',
        'ridge': 'ridge',
        'lasso': 'lasso',
        'random_forest_regressor': 'rf',
        'random_forest_classifier': 'rf',
        'gradient_boosting_regressor': 'gb',
        'gradient_boosting_classifier': 'gb',
        'svr': 'svr',
        'svm': 'svm',
        'knn_regressor': 'knn',
        'knn_classifier': 'knn',
        'adaboost_regressor': 'ada',
        'adaboost_classifier': 'ada',
        'logistic_regression': 'logistic'
    }

    prefix = model_prefixes.get(model_key, '')

    # Common parameter mappings
    param_mappings = {
        'n_estimators': ['n_estimators', f'{prefix}_n_estimators', f'{model_key}_n_estimators'],
        'max_depth': ['max_depth', f'{prefix}_max_depth', f'{model_key}_max_depth'],
        'min_samples_split': ['min_samples_split', f'{prefix}_min_samples_split'],
        'min_samples_leaf': ['min_samples_leaf', f'{prefix}_min_samples_leaf'],
        'learning_rate': ['learning_rate', f'{prefix}_learning_rate'],
        'C': ['C', f'{prefix}_C', 'svm_C'],
        'alpha': ['alpha', f'{prefix}_alpha'],
        'n_neighbors': ['n_neighbors', f'{prefix}_n_neighbors'],
        'kernel': ['kernel', f'{prefix}_kernel'],
        'gamma': ['gamma', f'{prefix}_gamma'],
        'epsilon': ['epsilon', f'{prefix}_epsilon'],
        'degree': ['degree', f'{prefix}_degree'],
        'subsample': ['subsample', f'{prefix}_subsample'],
        'max_features': ['max_features', f'{prefix}_max_features'],
        'criterion': ['criterion', f'{prefix}_criterion'],
        'penalty': ['penalty', f'{prefix}_penalty'],
        'solver': ['solver', f'{prefix}_solver'],
        'fit_intercept': ['fit_intercept', f'{prefix}_fit_intercept'],
        'tol': ['tol', f'{prefix}_tol'],
        'max_iter': ['max_iter', f'{prefix}_max_iter'],
        'bootstrap': ['bootstrap', f'{prefix}_bootstrap'],
        'weights': ['weights', f'{prefix}_weights'],
        'algorithm': ['algorithm', f'{prefix}_algorithm'],
        'loss': ['loss', f'{prefix}_loss'],
        'class_weight': ['class_weight', f'{prefix}_class_weight']
    }

    # Try to find each parameter
    for param_name, possible_keys in param_mappings.items():
        for key in possible_keys:
            if key in form_dict:
                value = form_dict[key]
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]

                 # Convert to appropriate type using helper
                value = convert_param_value(param_name, value)

                try:
                    # Convert to appropriate type
                    if param_name in ['n_estimators', 'max_depth', 'min_samples_split',
                                     'min_samples_leaf', 'degree', 'n_neighbors',
                                     'max_iter', 'leaf_size', 'cache_size']:
                        params[f"model__{param_name}"] = int(float(value))
                    elif param_name in ['C', 'alpha', 'learning_rate', 'epsilon',
                                       'tol', 'subsample', 'l1_ratio',
                                       'min_impurity_decrease']:
                        params[f"model__{param_name}"] = float(value)
                    elif param_name in ['fit_intercept', 'bootstrap', 'shrinking',
                                       'probability', 'warm_start', 'break_ties']:
                        params[f"model__{param_name}"] = str(value).lower() in ['true', '1', 'yes', 'on']
                    else:
                        params[f"model__{param_name}"] = value

                    # [DEBUG] print(f"  -> Found {param_name} as {key} = {value}")
                    break
                except Exception as e:
                    # [DEBUG] print(f"  -> Error converting {param_name}: {e}")
                    continue

    # Special handling for gamma in SVM/SVR
    if model_key in ['svr', 'svm'] and 'model__gamma' not in params:
        # Check if gamma was provided as scale/auto
        for key in ['gamma', 'svm_gamma', 'svr_gamma']:
            if key in form_dict:
                value = form_dict[key]
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]
                if value in ['scale', 'auto']:
                    params["model__gamma"] = value
                    # [DEBUG] print(f"  -> Set gamma to {value}")
                    break

    # [DEBUG] print(f"  Final params for {model_key}: {len(params)} parameters")
    return params

def get_dataset_profile(n_rows):
    if n_rows < 20_000:
        return "small"
    elif n_rows < 80_000:
        return "medium"
    else:
        return "large"

def get_model_keys_for_task(task_type, form_dict):
    """
    Get appropriate model keys for the task with user selection support
    """
    task_group = "regression" if task_type == "regression" else "classification"

    # Get all available models for the task
    all_models = list(MODEL_REGISTRY[task_group].keys())

    # Check if user selected specific models
    user_models = []
    if "models" in form_dict:
        models_value = form_dict["models"]
        if isinstance(models_value, list):
            user_models = models_value
        elif isinstance(models_value, str):
            user_models = [models_value]

    # Filter user models to only include valid ones
    valid_user_models = [m for m in user_models if m in all_models]

    # If user selected models, use them (filtered)
    if valid_user_models:
        return valid_user_models

    # Otherwise, use intelligent defaults based on task type
    if task_type == "regression":
        # Good regression models in order of preference
        preferred_models = [
            "random_forest_regressor",
            "gradient_boosting_regressor",
            "ridge",
            "linear_regression",
            "lasso",
            "knn_regressor",
            "svr"
        ]
    else:  # classification
        preferred_models = [
            "random_forest_classifier",
            "gradient_boosting_classifier",
            "logistic_regression",
            "knn_classifier",
            "svm"
        ]

    # Return only models that exist in registry
    return [m for m in preferred_models if m in all_models]

def fix_logistic_params(params):
    solver = params.get("model__solver", "lbfgs")
    penalty = params.get("model__penalty", "l2")

    # Invalid combinations → auto-fix
    if solver in ["lbfgs", "newton-cg"] and penalty in ["l1", "elasticnet"]:
        params["model__penalty"] = "l2"

    if solver == "liblinear" and penalty == "elasticnet":
        params["model__penalty"] = "l2"

    if penalty == "elasticnet" and solver != "saga":
        params["model__solver"] = "saga"

    return params

def generate_parameter_controls(model_key, params_config):
    """
    Generate HTML controls for model parameters
    """
    controls_html = ""

    # Group parameters by type for better organization
    basic_params = []
    advanced_params = []
    optimization_params = []

    for param in params_config:
        simple_name = param["simple_name"]

        # Categorize parameters
        if simple_name in ["n_estimators", "C", "alpha", "learning_rate", "max_depth",
                          "n_neighbors", "max_features", "subsample"]:
            basic_params.append(param)
        elif simple_name in ["random_state", "tol", "max_iter", "epsilon", "cache_size"]:
            optimization_params.append(param)
        else:
            advanced_params.append(param)

    # Generate controls for each category

    # Basic parameters
    if basic_params:
        controls_html += '<div class="parameter-category">'
        controls_html += '<h6><i class="fas fa-sliders-h me-2"></i>Basic Parameters</h6>'
        controls_html += '<div class="row g-3">'

        for param in basic_params:
            controls_html += generate_single_control(param, model_key)

        controls_html += '</div></div>'

    # Advanced parameters
    if advanced_params:
        controls_html += '<div class="parameter-category mt-4">'
        controls_html += '<h6><i class="fas fa-cogs me-2"></i>Advanced Parameters</h6>'
        controls_html += '<div class="row g-3">'

        for param in advanced_params:
            controls_html += generate_single_control(param, model_key)

        controls_html += '</div></div>'

    # Optimization parameters
    if optimization_params:
        controls_html += '<div class="parameter-category mt-4">'
        controls_html += '<h6><i class="fas fa-tachometer-alt me-2"></i>Optimization</h6>'
        controls_html += '<div class="row g-3">'

        for param in optimization_params:
            controls_html += generate_single_control(param, model_key)

        controls_html += '</div></div>'

    return controls_html

def generate_single_control(param, model_key):
    """
    Generate a single parameter control
    """
    ui_id = param["ui_id"]
    simple_name = param["simple_name"]
    param_type = param.get("type", "text")
    default_value = param.get("default", "")

    control_html = f'<div class="col-md-6"><div class="parameter-control">'

    # Label
    display_name = param.get("display_name", simple_name.replace("_", " ").title())
    control_html += f'<label class="form-label">{display_name}'
    control_html += f'<small class="text-muted ms-1">({param_type})</small>'
    control_html += '</label>'

    # Control based on type
    if param_type in ["int", "float"]:
        min_val = param.get("min", 0)
        max_val = param.get("max", 100)
        step = param.get("step", 1 if param_type == "int" else 0.01)

        control_html += f'''
        <div class="d-flex align-items-center gap-2 mb-2">
            <input type="range"
                   class="form-range"
                   id="{ui_id}"
                   min="{min_val}"
                   max="{max_val}"
                   step="{step}"
                   value="{default_value}"
                   oninput="document.getElementById('{ui_id}_value').textContent = this.value">
            <span class="badge bg-primary" id="{ui_id}_value">{default_value}</span>
        </div>
        <div class="d-flex justify-content-between small text-muted">
            <span>{min_val}</span>
            <span>Default: {default_value}</span>
            <span>{max_val}</span>
        </div>
        '''

    elif param_type == "categorical":
        options = param.get("options", param.get("values", []))
        control_html += f'<select class="form-select" id="{ui_id}">'
        for opt in options:
            selected = "selected" if opt == default_value else ""
            display_opt = "None" if opt is None else str(opt)
            control_html += f'<option value="{opt}" {selected}>{display_opt}</option>'
        control_html += '</select>'

    elif param_type == "bool":
        control_html += f'''
        <div class="form-check form-switch">
            <input class="form-check-input" type="checkbox" id="{ui_id}"
                   {"checked" if default_value else ""}>
            <label class="form-check-label" for="{ui_id}">
                {default_value and "Enabled" or "Disabled"}
            </label>
        </div>
        '''

    else:
        # Text input
        control_html += f'<input type="text" class="form-control" id="{ui_id}" value="{default_value}">'

    # Tooltip/help text
    if "description" in param:
        control_html += f'<small class="text-muted d-block mt-1">{param["description"]}</small>'

    control_html += '</div></div>'
    return control_html

def generate_regression_plots(y_train, y_train_pred, y_test, y_test_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io, base64

    sns.set(style="whitegrid")

    def _plot(y_true, y_pred, title):
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(title)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    return {
        "train_plot": _plot(y_train, y_train_pred, "Train: Actual vs Predicted"),
        "test_plot": _plot(y_test, y_test_pred, "Test: Actual vs Predicted")
    }

def save_regression_plots(y_true, y_pred, filename):

    plot_dir = os.path.join(app.root_path, "static", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    full_path = os.path.join(plot_dir, f"{filename}.png")

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.35)

    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)

    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{filename}: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(full_path, dpi=160)
    plt.close()

    return f"plots/{filename}.png"

def should_skip_svr(X, max_rows=8000):
    return len(X) > max_rows

def unified_feature_engineering(X, y=None, task_type="auto", target_col=None, mode="auto"):
    """
    PRODUCTION-GRADE unified feature engineering WITHOUT TIME SERIES
    Handles: Normal, Zero-inflated, Noisy data, High cardinality, Collinearity
    ENHANCED: Less aggressive for financial/stock data, better feature preservation
    """
    # [DEBUG] print(f"\n{'='*80}")
    # [DEBUG] print(f"UNIFIED FEATURE ENGINEERING - ENHANCED MODE (NON-TIME SERIES)")
    # [DEBUG] print(f"Mode: {mode}")
    # [DEBUG] print(f"Original shape: {X.shape}")
    # [DEBUG] print(f"Task type: {task_type}")
    # [DEBUG] print(f"Target column: {target_col}")
    # [DEBUG] print(f"{'='*80}")

    X_original = X.copy()

    # Initialize results dictionary
    fe_results = {
        "mode": mode,
        "original_shape": X.shape,
        "final_shape": None,
        "features_added": [],
        "features_removed": [],
        "transformations_applied": [],
        "warnings": [],
        "processing_steps": []
    }

    try:
        # ========== STEP 1: CRITICAL DATA VALIDATION ==========
        step_name = "validation"
        fe_results["processing_steps"].append({"name": step_name, "status": "in-progress"})
        # [DEBUG] print(f"\n1. CRITICAL DATA VALIDATION...")

        # Remove target column if accidentally included
        if target_col and target_col in X.columns:
            # [DEBUG] print(f"  ⚠️ Removing target column '{target_col}' from features to prevent data leakage")
            X = X.drop(columns=[target_col])
            fe_results["features_removed"].append(target_col)
            fe_results["warnings"].append(f"Target column '{target_col}' removed from features")

        # Remove pandas index columns - LESS AGGRESSIVE
        index_cols = []
        for col in X.columns:
            col_lower = str(col).lower()
            # Only remove obvious index columns
            if (col_lower.startswith('unnamed: 0') or
                col_lower == 'index' or
                col_lower == 'row_index' or
                col_lower == 'unnamed_0' or
                ('unnamed:' in col_lower and '_' not in col_lower.replace('unnamed:', '').replace(' ', ''))):
                index_cols.append(col)

        if index_cols:
            X = X.drop(columns=index_cols)
            fe_results["features_removed"].extend(index_cols)
            # [DEBUG] print(f"  → Removed {len(index_cols)} obvious index columns: {index_cols}")

        # Remove duplicate columns (exact duplicates)
        X = X.loc[:, ~X.columns.duplicated()]

        # Remove columns with too many missing values (>90%) - LESS AGGRESSIVE
        missing_ratios = X.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > 0.9].index.tolist()
        if high_missing_cols:
            X = X.drop(columns=high_missing_cols)
            fe_results["features_removed"].extend(high_missing_cols)
            fe_results["transformations_applied"].append("high_missing_removal")
            # [DEBUG] print(f"  → Removed {len(high_missing_cols)} columns with >90% missing values")

        # Check for constant columns but don't remove automatically - just warn
        constant_cols = []
        for col in X.columns:
            if X[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)

        if constant_cols:
            fe_results["warnings"].append(f"Constant columns detected: {constant_cols}. They may be removed later if problematic.")
            # [DEBUG] print(f"  ⚠️ Found {len(constant_cols)} constant columns: {constant_cols}")

        fe_results["processing_steps"][-1]["status"] = "completed"

        # ========== STEP 2: CHECK FOR IDENTICAL FEATURES (CRITICAL) ==========
        step_name = "duplicate_check"
        fe_results["processing_steps"].append({"name": step_name, "status": "in-progress"})
        # [DEBUG] print(f"\n2. CHECKING FOR IDENTICAL FEATURES...")

        # Check for perfectly identical columns
        columns_to_check = list(X.columns)
        removed_duplicates = []

        for i, col1 in enumerate(columns_to_check):
            if col1 not in X.columns:
                continue

            for col2 in columns_to_check[i+1:]:
                if col2 not in X.columns:
                    continue

                # Check if columns are identical
                try:
                    if X[col1].equals(X[col2]):
                        # [DEBUG] print(f"  ⚠️ Removing duplicate column: {col2} (identical to {col1})")
                        X = X.drop(columns=[col2])
                        removed_duplicates.append(col2)
                    else:
                        # Check correlation for numeric columns
                        if is_numeric_dtype(X[col1]) and is_numeric_dtype(X[col2]):
                            corr = X[col1].corr(X[col2])
                            if not np.isnan(corr) and abs(corr) > 0.999:  # Effectively identical
                                # [DEBUG] print(f"  ⚠️ Removing highly correlated column: {col2} (r={corr:.4f} with {col1})")
                                X = X.drop(columns=[col2])
                                removed_duplicates.append(col2)
                except:
                    pass

        if removed_duplicates:
            fe_results["features_removed"].extend(removed_duplicates)
            fe_results["transformations_applied"].append("duplicate_removal")
            # [DEBUG] print(f"  → Removed {len(removed_duplicates)} duplicate/identical columns")

        fe_results["processing_steps"][-1]["status"] = "completed"

        # ========== STEP 3: ROBUST IMPUTATION ==========
        step_name = "imputation"
        fe_results["processing_steps"].append({"name": step_name, "status": "in-progress"})
        # [DEBUG] print(f"\n3. ROBUST IMPUTATION...")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Add very small noise to prevent perfect collinearity issues
        if len(numeric_cols) > 0:
            np.random.seed(42)
            noise = np.random.normal(0, 1e-12, (len(X), len(numeric_cols)))  # Tiny noise
            X[numeric_cols] = X[numeric_cols] + noise
            # [DEBUG] print(f"  → Added minimal noise to {len(numeric_cols)} numeric columns to prevent collinearity")

        # Impute numeric columns
        imputed_numeric = []
        for col in numeric_cols:
            if X[col].isnull().any():
                # Handle infinite values first
                if np.any(np.isinf(X[col])):
                    finite_vals = X[col][np.isfinite(X[col])]
                    if len(finite_vals) > 0:
                        median_val = finite_vals.median()
                        X[col] = X[col].replace([np.inf, -np.inf], median_val)
                        # [DEBUG] print(f"    -> Replaced infinite values in '{col}' with median: {median_val}")

                # Now impute NaN
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col] = X[col].fillna(median_val)
                imputed_numeric.append(col)

        if imputed_numeric:
            fe_results["transformations_applied"].append(f"numeric_imputation:{len(imputed_numeric)}_cols")
            # [DEBUG] print(f"  → Imputed {len(imputed_numeric)} numeric columns with median")

        # Impute categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        imputed_cat = []
        for col in cat_cols:
            if X[col].isnull().any():
                mode_val = X[col].mode()
                if not mode_val.empty:
                    X[col] = X[col].fillna(mode_val.iloc[0])
                else:
                    X[col] = X[col].fillna('missing')
                imputed_cat.append(col)

        if imputed_cat:
            fe_results["transformations_applied"].append(f"categorical_imputation:{len(imputed_cat)}_cols")
            # [DEBUG] print(f"  → Imputed {len(imputed_cat)} categorical columns with mode")

        fe_results["processing_steps"][-1]["status"] = "completed"

        # ========== STEP 4: ENHANCED FEATURE ENGINEERING ==========
        step_name = "feature_engineering"
        fe_results["processing_steps"].append({"name": step_name, "status": "in-progress"})
        # [DEBUG] print(f"\n4. ENHANCED FEATURE ENGINEERING...")

        # Check for financial/stock data patterns
        financial_keywords = ['open', 'high', 'low', 'close', 'volume', 'price', 'adj', 'return', 'volatility']
        financial_features = []

        for col in X.columns:
            col_lower = str(col).lower()
            for keyword in financial_keywords:
                if keyword in col_lower:
                    financial_features.append(col)
                    break

        if financial_features:
            # [DEBUG] print(f"  → Detected {len(financial_features)} financial features: {financial_features[:5]}...")
            fe_results["warnings"].append(f"Financial data detected: {len(financial_features)} features. Using conservative feature engineering.")

        # Create simple interaction features if we have multiple numeric columns
        numeric_cols_final = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols_final) >= 2:
            # Create ratio features for the first few columns
            for i in range(min(3, len(numeric_cols_final))):
                for j in range(i+1, min(4, len(numeric_cols_final))):
                    col1, col2 = numeric_cols_final[i], numeric_cols_final[j]
                    try:
                        # Avoid division by zero or very small numbers
                        min_val2 = X[col2].abs().min()
                        if min_val2 > 1e-10:  # Not too close to zero
                            ratio_name = f"{col1}_div_{col2}"
                            X[ratio_name] = X[col1] / X[col2]
                            fe_results["features_added"].append(ratio_name)
                            # [DEBUG] print(f"    -> Created ratio feature: {ratio_name}")
                    except:
                        pass

        fe_results["processing_steps"][-1]["status"] = "completed"

        # ========== STEP 5: INTELLIGENT ENCODING ==========
        step_name = "encoding"
        fe_results["processing_steps"].append({"name": step_name, "status": "in-progress"})
        # [DEBUG] print(f"\n5. INTELLIGENT ENCODING...")

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if cat_cols:
            low_card_cols = []
            medium_card_cols = []
            high_card_cols = []

            for col in cat_cols:
                unique_count = X[col].nunique()
                if unique_count <= 15:  # Increased from 10
                    low_card_cols.append(col)
                elif unique_count <= 50:  # Increased from 20
                    medium_card_cols.append(col)
                else:
                    high_card_cols.append(col)

            # One-hot encode low cardinality
            if low_card_cols:
                dummies = pd.get_dummies(X[low_card_cols], prefix=low_card_cols, drop_first=True)
                X = pd.concat([X.drop(columns=low_card_cols), dummies], axis=1)
                fe_results["transformations_applied"].append(f"one_hot_encoding:{len(low_card_cols)}_cols")
                # [DEBUG] print(f"  → One-hot encoded {len(low_card_cols)} low-cardinality columns")

            # Frequency encode medium cardinality
            if medium_card_cols:
                for col in medium_card_cols:
                    freq = X[col].value_counts(normalize=True)
                    X[f"{col}_freq"] = X[col].map(freq)
                    fe_results["features_added"].append(f"{col}_freq")
                    fe_results["transformations_applied"].append(f"frequency_encoding:{col}")

                X = X.drop(columns=medium_card_cols)
                # [DEBUG] print(f"  → Frequency encoded {len(medium_card_cols)} medium-cardinality columns")

            # Target encoding for high cardinality if target is available
            if high_card_cols and y is not None:
                # [DEBUG] print(f"  → Attempting target encoding for {len(high_card_cols)} high-cardinality columns")
                for col in high_card_cols:
                    try:
                        # Calculate mean target per category
                        if isinstance(y, pd.Series):
                            target_means = y.groupby(X[col]).mean()
                            X[f"{col}_target_enc"] = X[col].map(target_means)
                            fe_results["features_added"].append(f"{col}_target_enc")
                            fe_results["transformations_applied"].append(f"target_encoding:{col}")
                    except:
                        pass

                # Drop original high cardinality columns after encoding
                X = X.drop(columns=high_card_cols)
                # [DEBUG] print(f"  → Target encoded {len(high_card_cols)} high-cardinality columns")
            elif high_card_cols:
                # If no target, use frequency encoding
                for col in high_card_cols:
                    freq = X[col].value_counts(normalize=True)
                    X[f"{col}_freq"] = X[col].map(freq)
                    fe_results["features_added"].append(f"{col}_freq")
                    fe_results["transformations_applied"].append(f"frequency_encoding:{col}")

                X = X.drop(columns=high_card_cols)
                # [DEBUG] print(f"  → Frequency encoded {len(high_card_cols)} high-cardinality columns (no target available)")

        fe_results["processing_steps"][-1]["status"] = "completed"

        # ========== STEP 6: COLLINEARITY DETECTION & REMOVAL ==========
        step_name = "collinearity"
        fe_results["processing_steps"].append({"name": step_name, "status": "in-progress"})
        # [DEBUG] print(f"\n6. COLLINEARITY DETECTION & REMOVAL...")

        numeric_cols_final = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols_final) > 1:
            # Calculate correlation matrix
            corr_matrix = X[numeric_cols_final].corr().abs()

            # Find highly correlated pairs (>0.98) - LESS AGGRESSIVE
            cols_to_remove = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.98:  # Increased threshold
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        # Keep the one with higher variance
                        if col1 not in cols_to_remove and col2 not in cols_to_remove:
                            var1 = X[col1].var()
                            var2 = X[col2].var()
                            if var1 >= var2:
                                cols_to_remove.add(col2)
                            else:
                                cols_to_remove.add(col1)

            if cols_to_remove:
                X = X.drop(columns=list(cols_to_remove))
                fe_results["features_removed"].extend(list(cols_to_remove))
                fe_results["transformations_applied"].append("collinearity_removal")
                # [DEBUG] print(f"  → Removed {len(cols_to_remove)} highly correlated features (r > 0.98)")

        fe_results["processing_steps"][-1]["status"] = "completed"

        # ========== STEP 7: FINAL VALIDATION & CLEANING ==========
        step_name = "final_validation"
        fe_results["processing_steps"].append({"name": step_name, "status": "in-progress"})
        # [DEBUG] print(f"\n7. FINAL VALIDATION & CLEANING...")

        # Ensure no infinite values
        numeric_cols_final = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols_final:
            if np.any(np.isinf(X[col])):
                finite_vals = X[col][np.isfinite(X[col])]
                if len(finite_vals) > 0:
                    median_val = finite_vals.median()
                    X[col] = X[col].replace([np.inf, -np.inf], median_val)
                else:
                    X[col] = X[col].replace([np.inf, -np.inf], 0)
                # [DEBUG] print(f"    -> Cleaned infinite values in '{col}'")

        # Ensure no NaN values remain
        if X.isnull().any().any():
            # Fill remaining NaN with appropriate values
            for col in X.columns:
                if X[col].isnull().any():
                    if is_numeric_dtype(X[col]):
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val)
                    else:
                        X[col] = X[col].fillna('missing')
            # [DEBUG] print(f"  → Filled remaining NaN values")

        # Remove any columns that became constant after all processing
        const_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
        if const_cols:
            X = X.drop(columns=const_cols)
            fe_results["features_removed"].extend(const_cols)
            # [DEBUG] print(f"  → Removed {len(const_cols)} columns that became constant after processing")

        # Final check: all values must be finite
        numeric_check = X.select_dtypes(include=[np.number])
        if not numeric_check.empty:
            assert np.all(np.isfinite(numeric_check)), "Non-finite values detected after final cleaning!"

        # Ensure we have at least 2 features
        if X.shape[1] < 2:
            # [DEBUG] print(f"  ⚠️ WARNING: Only {X.shape[1]} features remain. Adding safe features...")
            # Add safe features to prevent training issues
            for i in range(max(2, 3 - X.shape[1])):
                safe_feature_name = f"safe_feature_{i}"
                X[safe_feature_name] = np.random.normal(0, 1, len(X))
                fe_results["features_added"].append(safe_feature_name)
                fe_results["warnings"].append(f"Added safe feature {safe_feature_name} to prevent training issues")

        fe_results["processing_steps"][-1]["status"] = "completed"

        # ========== STORE FINAL RESULTS ==========
        fe_results["final_shape"] = X.shape
        fe_results["features_added_count"] = len(set(fe_results["features_added"]))
        fe_results["features_removed_count"] = len(set(fe_results["features_removed"]))  # FIXED: Changed from features_removed to fe_results["features_removed"]

        # [DEBUG] print(f"\n{'='*80}")
        # [DEBUG] print(f"FEATURE ENGINEERING COMPLETE")
        # [DEBUG] print(f"Original shape: {fe_results['original_shape']}")
        # [DEBUG] print(f"Final shape: {fe_results['final_shape']}")
        # [DEBUG] print(f"Features added: {fe_results['features_added_count']}")
        # [DEBUG] print(f"Features removed: {fe_results['features_removed_count']}")
        # [DEBUG] print(f"Transformations applied: {fe_results['transformations_applied']}")
        # [DEBUG] print(f"Warnings: {len(fe_results['warnings'])}")
        # [DEBUG] print(f"{'='*80}\n")

        return X, fe_results

    except Exception as e:
        error_msg = f"Feature engineering failed: {str(e)}"
        # [DEBUG] print(f"\n❌ ERROR in unified_feature_engineering: {error_msg}")
        import traceback
        traceback.print_exc()

        fe_results["warnings"].append(error_msg)
        for step in fe_results["processing_steps"]:
            if step["status"] == "in-progress":
                step["status"] = "failed"

        # Return original data if FE fails, with minimal cleaning
        # [DEBUG] print(f"  → Falling back to original data with minimal cleaning")
        X_fallback = X_original.copy()

        # Remove target column if present
        if target_col and target_col in X_fallback.columns:
            X_fallback = X_fallback.drop(columns=[target_col])

        # Remove obvious index columns
        index_cols = []
        for col in X_fallback.columns:
            col_lower = str(col).lower()
            if (col_lower.startswith('unnamed: 0') or
                col_lower == 'index' or
                col_lower == 'row_index'):
                index_cols.append(col)

        if index_cols:
            X_fallback = X_fallback.drop(columns=index_cols)

        # Basic imputation
        for col in X_fallback.select_dtypes(include=[np.number]).columns:
            if X_fallback[col].isnull().any():
                median_val = X_fallback[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_fallback[col] = X_fallback[col].fillna(median_val)

        fe_results["final_shape"] = X_fallback.shape
        fe_results["warnings"].append("Used fallback mode due to error")

        return X_fallback, fe_results

def improve_scaling_for_lasso(X, y, model_key):
    """Enhanced scaling for Lasso/Ridge to prevent convergence issues"""
    X_scaled = X.copy() if hasattr(X, 'copy') else X
    y_scaled = y.copy() if hasattr(y, 'copy') else y

    if model_key in ["lasso", "ridge"]:
        # [DEBUG] print(f"    -> Applying enhanced scaling for {model_key}")

        if isinstance(X_scaled, pd.DataFrame):
            numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                try:
                    # CRITICAL FIX: Use RobustScaler for Lasso, StandardScaler for Ridge
                    if model_key == "lasso":
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler(quantile_range=(25, 75))  # Less sensitive to outliers
                        # [DEBUG] print(f"    -> Using RobustScaler for Lasso (less sensitive to outliers)")
                    else:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        # [DEBUG] print(f"    -> Using StandardScaler for Ridge")

                    X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
                    # [DEBUG] print(f"    -> Scaled {len(numeric_cols)} numeric features")
                except Exception as e:
                    pass  # placeholder (debug print removed)
                    # [DEBUG] print(f"    -> Warning: Scaling failed: {e}")

        # Also scale target for regression
        if "regression" in model_key and isinstance(y_scaled, pd.Series):
            try:
                y_mean = y_scaled.mean()
                y_std = y_scaled.std()
                if y_std > 1e-10:
                    y_scaled = (y_scaled - y_mean) / y_std
                    # [DEBUG] print(f"    -> Standardized target (mean={y_mean:.2f}, std={y_std:.2f})")
            except Exception as e:
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"    -> Warning: Target scaling failed: {e}")

    return X_scaled, y_scaled

def validate_and_clean_training_data(X_data, y_data, model_key):
    """
    Validate and clean training data to prevent NaN/inf issues - COMPLETE FIXED VERSION
    With consistent target scaling for ALL regression models
    """
    X_clean = X_data.copy()
    y_clean = y_data.copy()

    # [DEBUG] print(f"  -> Data cleaning for {model_key}: Starting with shape {X_clean.shape}")

    # ========== STEP 1: CHECK IF TARGET ALREADY SCALED ==========
    is_regression_model = model_key in [
        "linear_regression", "ridge", "lasso",
        "random_forest_regressor", "gradient_boosting_regressor",
        "knn_regressor", "adaboost_regressor", "svr"
    ]

    # Skip automatic scaling if target already has a stored scaler
    if is_regression_model and model_key in DATASTORE.get('target_scalers', {}):
        # [DEBUG] print(f"  -> Target already scaled for {model_key}, skipping automatic scaling")
        # Still validate finite values
        if isinstance(y_clean, pd.Series):
            if y_clean.isna().any() or np.any(np.isinf(y_clean)):
                # [DEBUG] print(f"  -> Cleaning target values (NaN/Inf)")
                y_clean = y_clean.fillna(0).replace([np.inf, -np.inf], 0)
        # Continue with feature cleaning only
        # (the rest of the function will clean features)

    # ========== STEP 2: INITIAL TARGET SCALING CHECK (if not already scaled) ==========
    if is_regression_model and model_key not in DATASTORE.get('target_scalers', {}):
        # [DEBUG] print(f"  -> REGRESSION MODEL: Checking target scaling needs")

        # Calculate target statistics
        y_mean = y_clean.mean()
        y_std = y_clean.std()
        y_abs_max = y_clean.abs().max()
        y_abs_min = y_clean[y_clean != 0].abs().min() if (y_clean != 0).any() else 0

        # [DEBUG] print(f"    -> Target stats: mean={y_mean:.2e}, std={y_std:.2e}, max_abs={y_abs_max:.2e}")

        # Check if target needs robust scaling
        needs_scaling = (
            y_abs_max > 1e6 or  # Extremely large values
            y_std > 1e6 or      # High variance
            (y_abs_max > 0 and y_abs_min > 0 and y_abs_max / y_abs_min > 1e6)  # Large dynamic range
        )

        if needs_scaling:
            # [DEBUG] print(f"    -> WARNING: Target needs scaling for {model_key}")

            # Store original stats before scaling
            original_stats = {
                'mean': y_mean,
                'std': y_std,
                'min': y_clean.min(),
                'max': y_clean.max(),
                'median': y_clean.median()
            }

            # Apply RobustScaler (less sensitive to outliers)
            from sklearn.preprocessing import RobustScaler
            y_scaler = RobustScaler(quantile_range=(25, 75))
            y_clean_2d = y_clean.values.reshape(-1, 1)
            y_scaled = y_scaler.fit_transform(y_clean_2d).flatten()
            y_clean = pd.Series(y_scaled, index=y_clean.index)

            # Store scaler for consistent inverse transformation
            if 'target_scalers' not in DATASTORE:
                DATASTORE['target_scalers'] = {}
            DATASTORE['target_scalers'][model_key] = {
                'scaler': y_scaler,
                'original_stats': original_stats,
                'scaled': True
            }

            # Verify scaling worked
            y_scaled_mean = y_clean.mean()
            y_scaled_std = y_clean.std()
            # [DEBUG] print(f"    -> After scaling: mean={y_scaled_mean:.2e}, std={y_scaled_std:.2e}")

    # ========== STEP 3: MODEL-SPECIFIC PRE-SCALING ==========
    if model_key in ["lasso", "ridge"]:
        # [DEBUG] print(f"  -> PRE-SCALING: Applying enhanced scaling for {model_key}")
        X_clean, y_clean = improve_scaling_for_lasso(X_clean, y_clean, model_key)

    # ========== STEP 4: SPECIAL HANDLING FOR RIDGE REGRESSION ==========
    if model_key == "ridge":
        # [DEBUG] print(f"  -> SPECIAL: Enhanced cleaning for Ridge regression")

        # 1. Convert to DataFrame if needed
        if not isinstance(X_clean, pd.DataFrame):
            X_clean = pd.DataFrame(X_clean)

        # 2. Handle infinite values FIRST (causes most issues)
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in X_clean.columns:
                col_data = X_clean[col]

                # Replace infinite values
                if np.any(np.isinf(col_data)):
                    finite_vals = col_data[np.isfinite(col_data)]
                    if len(finite_vals) > 0:
                        # Use median and robust bounds
                        median_val = finite_vals.median()
                        q1 = finite_vals.quantile(0.25)
                        q3 = finite_vals.quantile(0.75)
                        iqr = q3 - q1

                        # Replace +inf with Q3 + 3*IQR, -inf with Q1 - 3*IQR
                        upper_bound = q3 + 3 * iqr
                        lower_bound = q1 - 3 * iqr

                        col_data = col_data.replace([np.inf], upper_bound)
                        col_data = col_data.replace([-np.inf], lower_bound)
                        X_clean[col] = col_data
                        # [DEBUG] print(f"    -> Replaced inf in {col}: +inf→{upper_bound:.2f}, -inf→{lower_bound:.2f}")

        # 3. Handle NaN values
        for col in numeric_cols:
            if col in X_clean.columns and X_clean[col].isna().any():
                # For Ridge, use mean instead of median (better for linear models)
                mean_val = X_clean[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0
                X_clean[col] = X_clean[col].fillna(mean_val)
                # [DEBUG] print(f"    -> Filled NaN in {col} with mean: {mean_val:.4f}")

        # 4. Check for constant columns (cause issues in Ridge)
        constant_cols = []
        for col in numeric_cols:
            if col in X_clean.columns:
                if X_clean[col].nunique() == 1:
                    constant_cols.append(col)

        if constant_cols:
            # Add small noise to constant columns instead of removing
            for col in constant_cols:
                noise = np.random.normal(0, 1e-10, len(X_clean))
                X_clean[col] = X_clean[col] + noise
                # [DEBUG] print(f"    -> Added noise to constant column {col}")

        # 5. Scale features to prevent large coefficient values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean[numeric_cols])
        X_clean[numeric_cols] = X_scaled
        # [DEBUG] print(f"    -> Standardized {len(numeric_cols)} numeric columns")

        # 6. Clean categorical columns (if any)
        cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col in X_clean.columns and X_clean[col].isna().any():
                mode_val = X_clean[col].mode()
                if not mode_val.empty:
                    X_clean[col] = X_clean[col].fillna(mode_val.iloc[0])
                else:
                    X_clean[col] = X_clean[col].fillna('missing')

        # 7. Additional target cleaning for Ridge
        if isinstance(y_clean, pd.Series):
            # Handle infinite values
            if np.any(np.isinf(y_clean)):
                finite_vals = y_clean[np.isfinite(y_clean)]
                if len(finite_vals) > 0:
                    median_val = finite_vals.median()
                    q1 = finite_vals.quantile(0.25)
                    q3 = finite_vals.quantile(0.75)
                    iqr = q3 - q1

                    upper_bound = q3 + 3 * iqr
                    lower_bound = q1 - 3 * iqr

                    y_clean = y_clean.replace([np.inf], upper_bound)
                    y_clean = y_clean.replace([-np.inf], lower_bound)

            # Handle NaN values
            if y_clean.isna().any():
                median_val = y_clean.median()
                if pd.isna(median_val):
                    median_val = 0
                y_clean = y_clean.fillna(median_val)

    # ========== STEP 5: SPECIAL HANDLING FOR LASSO REGRESSION ==========
    elif model_key == "lasso":
        # [DEBUG] print(f"  -> SPECIAL: Enhanced cleaning for Lasso regression")

        # 1. Convert to DataFrame if needed
        if not isinstance(X_clean, pd.DataFrame):
            X_clean = pd.DataFrame(X_clean)

        # 2. Handle infinite values
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in X_clean.columns:
                col_data = X_clean[col]

                # Replace infinite values
                if np.any(np.isinf(col_data)):
                    finite_vals = col_data[np.isfinite(col_data)]
                    if len(finite_vals) > 0:
                        median_val = finite_vals.median()
                        q1 = finite_vals.quantile(0.25)
                        q3 = finite_vals.quantile(0.75)
                        iqr = q3 - q1

                        # Cap extreme values for Lasso stability
                        upper_bound = q3 + 1.5 * iqr
                        lower_bound = q1 - 1.5 * iqr

                        col_data = col_data.replace([np.inf], upper_bound)
                        col_data = col_data.replace([-np.inf], lower_bound)
                        X_clean[col] = col_data
                        # [DEBUG] print(f"    -> Capped extreme values in {col}")

        # 3. Feature scaling for Lasso
        from sklearn.preprocessing import RobustScaler

        # Only scale if not already scaled
        if not hasattr(X_clean, '_already_scaled') and len(numeric_cols) > 0:
            scaler = RobustScaler(quantile_range=(25, 75))
            X_scaled = scaler.fit_transform(X_clean[numeric_cols])
            X_clean[numeric_cols] = X_scaled
            # [DEBUG] print(f"    -> Robust-scaled {len(numeric_cols)} numeric columns for Lasso")
            X_clean._already_scaled = True

        # 4. Add regularization to prevent perfect collinearity
        if len(numeric_cols) > 1:
            np.random.seed(42)
            tiny_noise = np.random.normal(0, 1e-8, X_clean[numeric_cols].shape)
            X_clean[numeric_cols] = X_clean[numeric_cols] + tiny_noise
            # [DEBUG] print(f"    -> Added minimal noise to break perfect collinearity")

        # 5. Additional target validation for Lasso
        if isinstance(y_clean, pd.Series):
            # Check if target was already scaled in step 1
            if model_key in DATASTORE.get('target_scalers', {}):
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"    -> Target already scaled for Lasso")
            else:
                # Ensure target is reasonable for Lasso
                y_var = y_clean.var()
                if y_var > 1e9:
                    pass  # placeholder (debug print removed)
                    # [DEBUG] print(f"    -> WARNING: High target variance for Lasso ({y_var:.2e})")

    # ========== STEP 6: REGULAR CLEANING FOR OTHER MODELS ==========
    else:
        # [DEBUG] print(f"  -> Regular cleaning for {model_key}")

        if isinstance(X_clean, pd.DataFrame):
            # Get initial NaN count
            nan_count_before = X_clean.isna().sum().sum()
            inf_count_before = np.isinf(X_clean.select_dtypes(include=[np.number])).sum().sum()

            if nan_count_before > 0 or inf_count_before > 0:
                # [DEBUG] print(f"  -> Found {nan_count_before} NaN and {inf_count_before} inf values, cleaning...")

                # Handle infinite values first
                numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in X_clean.columns:
                        col_data = X_clean[col]

                        # Replace infinite values
                        if np.any(np.isinf(col_data)):
                            finite_vals = col_data[np.isfinite(col_data)]
                            if len(finite_vals) > 0:
                                median_val = finite_vals.median()
                                std_val = finite_vals.std()

                                # Replace +inf with median + 3*std
                                col_data = col_data.replace([np.inf], median_val + 3 * std_val)
                                # Replace -inf with median - 3*std
                                col_data = col_data.replace([-np.inf], median_val - 3 * std_val)
                                X_clean[col] = col_data

                # Handle NaN values
                for col in numeric_cols:
                    if col in X_clean.columns and X_clean[col].isna().any():
                        median_val = X_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0
                        X_clean[col] = X_clean[col].fillna(median_val)

                # Clean categorical columns
                cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns
                for col in cat_cols:
                    if col in X_clean.columns and X_clean[col].isna().any():
                        mode_val = X_clean[col].mode()
                        if not mode_val.empty:
                            X_clean[col] = X_clean[col].fillna(mode_val.iloc[0])
                        else:
                            X_clean[col] = X_clean[col].fillna('missing')

        # Clean target variable if it's a pandas Series
        if isinstance(y_clean, pd.Series):
            # Check if target was already scaled in step 1
            if not (model_key in DATASTORE.get('target_scalers', {}) and
                   DATASTORE['target_scalers'][model_key].get('scaled', False)):

                # Handle infinite values in target
                if np.any(np.isinf(y_clean)):
                    finite_vals = y_clean[np.isfinite(y_clean)]
                    if len(finite_vals) > 0:
                        median_val = finite_vals.median()
                        std_val = finite_vals.std()
                        y_clean = y_clean.replace([np.inf], median_val + 3 * std_val)
                        y_clean = y_clean.replace([-np.inf], median_val - 3 * std_val)

                # Handle NaN in target
                if y_clean.isna().any():
                    median_val = y_clean.median()
                    y_clean = y_clean.fillna(median_val)

    # ========== STEP 7: FINAL VALIDATION AND CLEANUP ==========
    if isinstance(X_clean, pd.DataFrame):
        nan_count_after = X_clean.isna().sum().sum()
        inf_count_after = np.isinf(X_clean.select_dtypes(include=[np.number])).sum().sum()

        if nan_count_after > 0 or inf_count_after > 0:
            # [DEBUG] print(f"  -> WARNING: Still found {nan_count_after} NaN and {inf_count_after} inf after cleaning!")
            # [DEBUG] print(f"  -> Forcing final cleanup...")

            # Force fill any remaining NaN
            X_clean = X_clean.fillna(0)

            # Force replace any remaining Inf
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in X_clean.columns:
                    X_clean[col] = X_clean[col].replace([np.inf, -np.inf], 0)

    if isinstance(y_clean, pd.Series):
        if y_clean.isna().any() or np.any(np.isinf(y_clean)):
            # [DEBUG] print(f"  -> Forcing final cleanup of target...")
            y_clean = y_clean.fillna(0).replace([np.inf, -np.inf], 0)

    # ========== STEP 8: FINAL STATS AND VERIFICATION ==========
    # [DEBUG] print(f"  -> Data cleaning complete. Final shape: {X_clean.shape}")

    # Log final statistics
    if isinstance(X_clean, pd.DataFrame):
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            x_mean = X_clean[numeric_cols].mean().mean()
            x_std = X_clean[numeric_cols].std().mean()
            # [DEBUG] print(f"  -> X stats: mean={x_mean:.4f}, std={x_std:.4f}")

    if isinstance(y_clean, pd.Series):
        y_mean = y_clean.mean()
        y_std = y_clean.std()
        # [DEBUG] print(f"  -> y stats: mean={y_mean:.4f}, std={y_std:.4f}")

        # Verify target is reasonable
        if is_regression_model and abs(y_mean) > 1000:
            pass  # placeholder (debug print removed)
            # [DEBUG] print(f"  -> WARNING: Target mean is large ({y_mean:.2f}) - may cause convergence issues")

    return X_clean, y_clean

def calculate_consistent_regression_metrics(y_true, y_pred, model_key=None):
    """
    Calculate regression metrics with proper inverse scaling
    Returns metrics in ORIGINAL scale for meaningful comparison
    """
    # Convert to numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Remove NaN and Inf
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "rmse_scaled": np.nan}

    # Try to get original scale metrics
    if model_key and 'target_scalers' in DATASTORE and model_key in DATASTORE['target_scalers']:
        scaler_info = DATASTORE['target_scalers'][model_key]
        if 'scaler' in scaler_info and scaler_info.get('scaled', False):
            try:
                # Inverse transform to original scale
                scaler = scaler_info['scaler']
                y_true_orig = scaler.inverse_transform(y_true_clean.reshape(-1, 1)).flatten()
                y_pred_orig = scaler.inverse_transform(y_pred_clean.reshape(-1, 1)).flatten()

                # Calculate metrics in original scale
                r2 = r2_score(y_true_orig, y_pred_orig)
                rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
                mae = mean_absolute_error(y_true_orig, y_pred_orig)

                # Also calculate scaled metrics for reference
                rmse_scaled = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

                return {
                    "r2": r2,
                    "rmse": rmse,  # Original scale RMSE
                    "rmse_scaled": rmse_scaled,  # Scaled RMSE
                    "mae": mae
                }
            except Exception as e:
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"  -> Could not inverse transform: {e}")

    # Fallback: calculate metrics as-is
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)

    return {
        "r2": r2,
        "rmse": rmse,
        "rmse_scaled": rmse,
        "mae": mae
    }

def build_lasso_specific_pipeline(model_cls, X):
    """
    Build optimized pipeline specifically for Lasso regression
    with enhanced convergence settings
    """
    numeric_cols, categorical_cols = split_columns(X)

    # [DEBUG] print(f"  -> Building Lasso-specific pipeline with convergence optimizations")

    # Use RobustScaler for Lasso (better with outliers)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(quantile_range=(25, 75)))  # Robust scaling
    ])

    # Create preprocessor
    transformers = [("num", numeric_pipeline, numeric_cols)]

    if categorical_cols:
        # For Lasso, use simpler categorical encoding
        usable_cat_cols = []
        for col in categorical_cols:
            if X[col].nunique() <= 10:  # Only low cardinality
                usable_cat_cols.append(col)

        if usable_cat_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    drop='first'  # Avoid dummy variable trap
                ))
            ])
            transformers.append(("cat", categorical_pipeline, usable_cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # Create Lasso model with convergence-optimized defaults
    lasso_model = model_cls(
        alpha=1.0,
        max_iter=5000,  # High iteration count
        tol=1e-4,       # Reasonable tolerance
        selection='cyclic',  # Better convergence
        random_state=42,
        warm_start=False  # Don't use warm start for grid search
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", lasso_model)
    ])

def build_optimized_pipeline(model_cls, X, model_key):
    """
    Build optimized pipeline based on model type with robust NaN handling
    Specialized for Ridge regression
    """
    numeric_cols, categorical_cols = split_columns(X)

    if model_key == "lasso":
        return build_lasso_specific_pipeline(model_cls,X)
    # SPECIAL HANDLING FOR RIDGE - COMPLETELY REVISED

    if model_key == "ridge":
        # [DEBUG] print(f"  -> Building robust pipeline for Ridge regression")

        # 1. Ultra-robust numeric preprocessing for Ridge
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", fill_value=0)),
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ])

        # 2. Conservative Ridge configuration
        # Use 'cholesky' solver - most stable for well-conditioned problems
        # or 'svd' for singular matrices
        ridge_params = {
            "alpha": 1.0,  # Default regularization
            "fit_intercept": True,
            "copy_X": True,
            "max_iter": 5000,  # Increased from default
            "tol": 1e-6,  # Tighter tolerance
            "solver": 'auto',  # Let sklearn choose best solver
            "random_state": 42,
            "positive": False  # Don't force positive coefficients
        }

        model_instance = model_cls(**ridge_params)

        # [DEBUG] print(f"  -> Ridge configured with: {ridge_params}")

    # SPECIAL HANDLING FOR LASSO
    elif model_key == "lasso":
        # [DEBUG] print(f"  -> Using enhanced pipeline for Lasso")

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(5, 95)))
        ])
        model_instance = model_cls()

    # Other linear models
    elif model_key in ["linear_regression", "ridge", "lasso", "logistic_regression"]:
        # [DEBUG] print(f"  -> Using standard pipeline for {model_key}")

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        model_instance = model_cls()

    else:
        # For non-linear models, use standard pipeline
        return build_pipeline(model_cls, X)

    # COMMON CATEGORICAL HANDLING
    if categorical_cols:
        usable_cat_cols = []
        for col in categorical_cols:
            non_null_count = X[col].notna().sum()
            unique_count = X[col].nunique()

            if non_null_count > 10 and unique_count <= 20:
                usable_cat_cols.append(col)
            else:
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"  -> Skipping column {col}: {non_null_count} non-null, {unique_count} unique")

        if usable_cat_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    max_categories=20
                ))
            ])

            transformers = [
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, usable_cat_cols)
            ]
        else:
            transformers = [("num", numeric_pipeline, numeric_cols)]
    else:
        transformers = [("num", numeric_pipeline, numeric_cols)]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_instance)
    ])

def validate_data_for_ridge(X, y):
    """
    Specialized data validation for Ridge regression
    Returns cleaned data and diagnostics
    """
    X_clean = X.copy()
    y_clean = y.copy()

    # [DEBUG] print(f"  === RIDGE DATA VALIDATION ===")

    # 1. Convert to DataFrame if needed
    if not isinstance(X_clean, pd.DataFrame):
        X_clean = pd.DataFrame(X_clean)

    # 2. Handle numeric columns
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in X_clean.columns:
            # Replace infinite values with NaN first
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)

    # 3. Check for constant columns
    constant_cols = []
    for col in numeric_cols:
        if col in X_clean.columns:
            non_null = X_clean[col].dropna()
            if len(non_null) > 0 and non_null.nunique() <= 1:
                constant_cols.append(col)
                # Add tiny noise to constant columns
                noise = np.random.normal(0, 1e-8, len(X_clean))
                X_clean[col] = X_clean[col] + noise
                # [DEBUG] print(f"    -> Added noise to constant column: {col}")

    # 4. Check condition number (if not too large)
    if len(numeric_cols) > 0 and len(X_clean) > 0:
        X_numeric = X_clean[numeric_cols].fillna(0).values
        if X_numeric.shape[1] > 0:
            # Add tiny regularization to prevent singular matrix
            X_numeric = X_numeric + np.random.normal(0, 1e-8, X_numeric.shape)

    # 5. Final imputation
    for col in numeric_cols:
        if col in X_clean.columns and X_clean[col].isna().any():
            mean_val = X_clean[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            X_clean[col] = X_clean[col].fillna(mean_val)

    # 6. Validate target
    if isinstance(y_clean, pd.Series):
        # Handle infinite values
        y_clean = y_clean.replace([np.inf, -np.inf], np.nan)

        # Check for constant target
        if y_clean.nunique() <= 1:
            # [DEBUG] print(f"    -> WARNING: Target has {y_clean.nunique()} unique value(s)")
            # Add tiny noise to prevent issues
            if y_clean.nunique() == 1:
                y_clean = y_clean + np.random.normal(0, 1e-8, len(y_clean))

        # Fill NaN
        if y_clean.isna().any():
            median_val = y_clean.median()
            if pd.isna(median_val):
                median_val = 0
            y_clean = y_clean.fillna(median_val)

    # 7. Check for perfect collinearity
    if len(numeric_cols) > 1:
        corr_matrix = X_clean[numeric_cols].corr().abs()
        # Identify highly correlated pairs (>0.999)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.999:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2))

        if high_corr_pairs:
            # [DEBUG] print(f"    -> Found {len(high_corr_pairs)} highly correlated pairs")
            # Keep first column, add noise to second
            for col1, col2 in high_corr_pairs[:3]:  # Limit to first 3
                if col2 in X_clean.columns:
                    noise = np.random.normal(0, 1e-8, len(X_clean))
                    X_clean[col2] = X_clean[col2] + noise
                    # [DEBUG] print(f"    -> Added noise to {col2} (correlated with {col1})")

    # [DEBUG] print(f"  === END RIDGE VALIDATION ===")

    return X_clean, y_clean

def train_ridge_safely(pipeline, X, y, scoring="r2", cv=5):
    """
    Safe Ridge training with comprehensive error handling
    """
    try:
        # [DEBUG] print(f"  -> Attempting safe Ridge training...")

        # Validate data first
        X_valid, y_valid = validate_data_for_ridge(X, y)

        # Check for any remaining issues
        if isinstance(X_valid, pd.DataFrame):
            if X_valid.isna().any().any():
                # [DEBUG] print(f"  -> WARNING: Still have NaN in X after validation")
                X_valid = X_valid.fillna(0)

        if isinstance(y_valid, pd.Series):
            if y_valid.isna().any():
                # [DEBUG] print(f"  -> WARNING: Still have NaN in y after validation")
                y_valid = y_valid.fillna(0)

        # Try different solvers if default fails
        solvers_to_try = ['auto', 'cholesky', 'svd', 'lsqr', 'sag']
        best_model = None
        best_score = -np.inf

        for solver in solvers_to_try:
            try:
                # [DEBUG] print(f"  -> Trying Ridge with solver='{solver}'...")

                # Update pipeline with current solver
                pipeline.named_steps['model'].set_params(solver=solver)

                # Fit the model
                pipeline.fit(X_valid, y_valid)

                # Cross-validation score
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(
                    pipeline, X_valid, y_valid,
                    cv=min(cv, len(y_valid)//10),  # Ensure enough samples per fold
                    scoring=scoring,
                    n_jobs=1,  # Single job for stability
                    error_score='raise'
                )

                score = np.mean(cv_scores)
                # [DEBUG] print(f"    -> Solver '{solver}': CV score = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = pipeline

            except Exception as solver_error:
                # [DEBUG] print(f"    -> Solver '{solver}' failed: {str(solver_error)[:100]}")
                continue

        if best_model is not None:
            # [DEBUG] print(f"  -> Best Ridge model: score = {best_score:.4f}")
            return best_model, best_score
        else:
            # [DEBUG] print(f"  -> All solvers failed, using fallback")
            # Fallback: simple linear regression
            from sklearn.linear_model import LinearRegression
            fallback_pipeline = Pipeline([
                ('preprocessor', pipeline.named_steps['preprocessor']),
                ('model', LinearRegression())
            ])
            fallback_pipeline.fit(X_valid, y_valid)
            return fallback_pipeline, 0.0

    except Exception as e:
        # [DEBUG] print(f"  -> Ridge training failed: {str(e)[:100]}")
        # Return a dummy model
        return None, np.nan

def get_cv_strategy(model_key, y):
    """
    Returns proper CV strategy (standard cross-validation without time series)
    """
    n_samples = len(y)

    # Determine appropriate number of folds based on dataset size
    if n_samples < 1000:
        n_splits = min(5, max(3, n_samples // 100))
    elif n_samples < 10000:
        n_splits = 5
    elif n_samples < 50000:
        n_splits = 3
    else:
        n_splits = 2

    n_splits = max(2, n_splits)  # At least 2 folds
    # [DEBUG] print(f"  -> Using Standard {n_splits}-fold CV")

    return n_splits

def fix_logistic_regression_param_grid(param_grid):
    """
    Fix logistic regression parameter grid to ensure valid combinations
    OPTIMIZED: Preserves more parameters for manual mode
    """
    if not param_grid:
        return param_grid

    # [DEBUG] print(f"  -> Optimizing logistic regression parameter grid")
    # [DEBUG] print(f"  -> Original keys: {list(param_grid.keys())}")

    # Check if we're in manual or auto mode
    tuning_mode = DATASTORE.get("tuning_mode", "auto")

    if tuning_mode == "auto":
        # AUTO MODE: Drastically reduce for speed
        # [DEBUG] print(f"  -> AUTO MODE: Drastically reducing grid")

        tiny_grid = {}

        # 1. C parameter: Use only 2 values
        for key in param_grid.keys():
            if 'C' in key:
                if len(param_grid[key]) > 2:
                    tiny_grid[key] = [0.1, 1.0]
                else:
                    tiny_grid[key] = param_grid[key]
                # [DEBUG] print(f"  -> C: {len(param_grid[key])} → {len(tiny_grid[key])} values")

        # 2. Solver: Use ONLY 'lbfgs' (fastest and most stable)
        for key in param_grid.keys():
            if 'solver' in key:
                tiny_grid[key] = ['lbfgs']
                # [DEBUG] print(f"  -> Solver: {len(param_grid[key])} → 1 value ('lbfgs')")

        # 3. Penalty: Use ONLY 'l2' (compatible with lbfgs)
        for key in param_grid.keys():
            if 'penalty' in key:
                tiny_grid[key] = ['l2']
                # [DEBUG] print(f"  -> Penalty: {len(param_grid[key])} → 1 value ('l2')")

        # 4. Max iterations: Use only 1 value
        for key in param_grid.keys():
            if 'max_iter' in key:
                tiny_grid[key] = [1000]
                # [DEBUG] print(f"  -> Max_iter: {len(param_grid[key])} → 1 value")

        # 5. Tolerance: Use only 1 value
        for key in param_grid.keys():
            if 'tol' in key:
                tiny_grid[key] = [1e-4]
                # [DEBUG] print(f"  -> Tol: {len(param_grid[key])} → 1 value")

        # 6. REMOVE l1_ratio and class_weight for speed in auto mode
        for key in param_grid.keys():
            if 'l1_ratio' in key or 'class_weight' in key:
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"  -> Removing {key} for speed (auto mode)")
                # Don't add to tiny_grid

    else:
        # MANUAL MODE: Keep more parameters
        # [DEBUG] print(f"  -> MANUAL MODE: Preserving more parameters")

        tiny_grid = param_grid.copy()

        # Still ensure valid combinations
        for key in param_grid.keys():
            if 'solver' in key and 'penalty' in key:
                solver = param_grid.get(key, ['lbfgs'])[0]
                penalty = param_grid.get(key.replace('solver', 'penalty'), ['l2'])[0]

                # Fix invalid combinations
                if solver == 'lbfgs' and penalty == 'l1':
                    tiny_grid[key.replace('solver', 'penalty')] = ['l2']
                    # [DEBUG] print(f"  -> Fixed: solver='lbfgs' incompatible with penalty='l1', changed to 'l2'")

        # Ensure at least C parameter exists
        if not any('C' in key for key in tiny_grid.keys()):
            tiny_grid['model__C'] = [0.1, 1.0, 10.0]
            # [DEBUG] print(f"  -> Added C parameter: {tiny_grid['model__C']}")

    # Calculate total combinations
    total_combos = 1
    for values in tiny_grid.values():
        total_combos *= len(values)

    # [DEBUG] print(f"  -> FINAL: {total_combos} parameter combinations")
    # [DEBUG] print(f"  -> Final keys: {list(tiny_grid.keys())}")

    return tiny_grid

def robust_hyperparameter_search(search_obj, X_tr, y_tr, model_key, param_grid):
    """
    Robust hyperparameter search with COMPREHENSIVE data cleaning
    WITH SPECIAL HANDLING FOR RIDGE NaN ISSUES
    """
    try:
        # [DEBUG] print(f"  -> Attempting hyperparameter search for {model_key}")

        # ================= CLEAN DATA FIRST =================
        X_tr_clean, y_tr_clean = validate_and_clean_training_data(X_tr, y_tr, model_key)

        # SPECIAL HANDLING FOR RIDGE
        if model_key == "ridge":
            # [DEBUG] print(f"  -> SPECIAL: Additional Ridge-specific data validation")

            # Check for constant columns
            constant_cols = []
            if isinstance(X_tr_clean, pd.DataFrame):
                for col in X_tr_clean.columns:
                    if X_tr_clean[col].nunique() <= 1:
                        constant_cols.append(col)

            if constant_cols:
                # [DEBUG] print(f"  -> WARNING: Found constant columns in Ridge training: {constant_cols}")
                # Add tiny noise to constant columns
                for col in constant_cols:
                    X_tr_clean[col] = X_tr_clean[col] + np.random.normal(0, 1e-10, len(X_tr_clean))

        # Final NaN check and fill
        if isinstance(X_tr_clean, pd.DataFrame):
            if X_tr_clean.isna().any().any():
                # [DEBUG] print(f"  -> Filling remaining NaN in X with 0")
                X_tr_clean = X_tr_clean.fillna(0)

        if isinstance(y_tr_clean, pd.Series):
            if y_tr_clean.isna().any():
                # [DEBUG] print(f"  -> Filling remaining NaN in y with median")
                y_tr_clean = y_tr_clean.fillna(y_tr_clean.median() if y_tr_clean.notna().any() else 0)

        # [DEBUG] print(f"  -> Final data check: X NaN={X_tr_clean.isna().sum().sum() if hasattr(X_tr_clean, 'isna') else 0}, "
              # [DEBUG] f"y NaN={y_tr_clean.isna().sum() if hasattr(y_tr_clean, 'isna') else 0}")

        # ================= SPECIAL HANDLING FOR LOGISTIC REGRESSION =================
        if model_key == "logistic_regression":
            # [DEBUG] print(f"  -> SPECIAL: Logistic regression search with enhanced settings")

            # Simplify parameter grid
            if hasattr(search_obj, 'param_grid'):
                search_obj.param_grid = fix_logistic_regression_param_grid(search_obj.param_grid)

            # Reduce CV folds for faster training
            if hasattr(search_obj, 'cv') and search_obj.cv > 3:
                search_obj.cv = 3
                # [DEBUG] print(f"  -> Reduced CV folds to 3 for faster training")

            # For large datasets, use single job
            if len(X_tr_clean) > 10000:
                search_obj.n_jobs = 1
                # [DEBUG] print(f"  -> Set n_jobs=1 for memory stability")

        # ================= ATTEMPT HYPERPARAMETER SEARCH =================
        # [DEBUG] print(f"  -> Fitting search with {len(X_tr_clean)} samples...")
        search_start = time.time()

        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)

            try:
                # SPECIAL HANDLING FOR RIDGE WITH NAN ISSUES
                if model_key == "ridge":
                    # [DEBUG] print(f"  -> SPECIAL: Using safe Ridge fitting with fallback")

                    try:
                        search_obj.fit(X_tr_clean, y_tr_clean)
                    except Exception as ridge_error:
                        # [DEBUG] print(f"  -> Ridge search failed: {ridge_error}")

                        # Try fallback with only stable solvers
                        # [DEBUG] print(f"  -> Trying fallback with only 'cholesky' solver")

                        # Create a fallback search with only cholesky solver
                        fallback_grid = param_grid.copy()
                        for key in fallback_grid:
                            if 'solver' in key:
                                fallback_grid[key] = ['cholesky']

                        if hasattr(search_obj, '__class__'):
                            # Create new search object with fallback grid
                            fallback_search = search_obj.__class__(
                                estimator=search_obj.estimator,
                                param_grid=fallback_grid,
                                cv=min(3, search_obj.cv),
                                scoring=search_obj.scoring,
                                n_jobs=1,
                                error_score='raise'
                            )

                            fallback_search.fit(X_tr_clean, y_tr_clean)
                            search_obj = fallback_search
                            # [DEBUG] print(f"  -> Fallback search succeeded with 'cholesky' solver")
                        else:
                            raise ridge_error
                else:
                    # Normal fitting for other models
                    search_obj.fit(X_tr_clean, y_tr_clean)

            except Exception as fit_error:
                # [DEBUG] print(f"  -> Fit failed: {fit_error}")

                # Try with even simpler approach
                # [DEBUG] print(f"  -> Trying fallback: fitting without hyperparameter search")
                try:
                    # Create simple pipeline as fallback
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler

                    if model_key == "ridge":
                        from sklearn.linear_model import Ridge
                        fallback_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', Ridge(
                                alpha=1.0,
                                solver='cholesky',  # Most stable
                                max_iter=10000,
                                random_state=42
                            ))
                        ])
                    elif model_key == "logistic_regression":
                        from sklearn.linear_model import LogisticRegression
                        fallback_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', LogisticRegression(
                                solver='lbfgs',
                                max_iter=1000,
                                random_state=42,
                                n_jobs=1
                            ))
                        ])
                    elif "regression" in model_key.lower():
                        from sklearn.linear_model import LinearRegression
                        fallback_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', LinearRegression())
                        ])
                    else:
                        # Default fallback
                        fallback_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', search_obj.estimator.named_steps['model'].__class__())
                        ])

                    fallback_pipeline.fit(X_tr_clean, y_tr_clean)

                    # Create mock search result
                    class MockSearchResult:
                        def __init__(self, pipeline):
                            self.best_estimator_ = pipeline
                            self.best_score_ = 0.5 if "classification" in model_key else 0.0
                            self.best_params_ = {'model__dummy': 'fallback'}

                    search_obj = MockSearchResult(fallback_pipeline)
                    # [DEBUG] print(f"  -> Fallback fitting succeeded")

                except Exception as e2:
                    # [DEBUG] print(f"  -> Fallback also failed: {e2}")
                    return None, f"all_attempts_failed"

        search_time = time.time() - search_start

        if hasattr(search_obj, 'best_estimator_') and search_obj.best_estimator_ is not None:
            # [DEBUG] print(f"  -> Search successful in {search_time:.1f}s with best score: {search_obj.best_score_:.4f}")
            return search_obj, "success"
        else:
            # [DEBUG] print(f"  -> Search completed but no best estimator found")
            return None, "no_best_estimator"

    except Exception as e:
        error_msg = str(e)
        # [DEBUG] print(f"  -> Hyperparameter search failed: {error_msg[:100]}")
        return None, f"search_failed: {error_msg[:50]}"

def check_and_fix_collinearity(X):
    """
    Check for perfect collinearity and fix by removing highly correlated features
    """
    X_clean = X.copy()
    removed_cols = []

    if len(X_clean.columns) > 1:
        # Calculate correlation matrix
        corr_matrix = X_clean.corr().abs()

        # Find pairs with near-perfect correlation
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.99:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

        # Remove one column from each highly correlated pair
        cols_to_remove = set()
        for col1, col2, corr_val in high_corr_pairs:
            if col1 not in cols_to_remove and col2 not in cols_to_remove:
                # Keep the column with higher variance
                var1 = X_clean[col1].var()
                var2 = X_clean[col2].var()

                if var1 >= var2:
                    cols_to_remove.add(col2)
                else:
                    cols_to_remove.add(col1)

        if cols_to_remove:
            X_clean = X_clean.drop(columns=list(cols_to_remove))
            removed_cols = list(cols_to_remove)
            # [DEBUG] print(f"  -> Removed {len(removed_cols)} columns due to perfect collinearity: {removed_cols}")

    return X_clean, removed_cols

# ============================================================================
# FIXED ZERO‑INFLATED TRAINING FUNCTION
# ============================================================================
def train_zero_inflated_models(X, target_info, selected_features, form_data):
    """
    Two-stage training for zero-inflated targets - COMPLETE FIXED VERSION
    """
    global TRAINING_STATUS, DATASTORE

    # [DEBUG] print(f"\n{'='*80}")
    # [DEBUG] print("TRAIN_ZERO_INFLATED_MODELS: Starting")
    # [DEBUG] print(f"DEBUG: target_info type: {type(target_info)}")

    start_time = time.time()

    # FIX: Handle different types of target_info
    if isinstance(target_info, dict):
        # [DEBUG] print(f"✓ target_info is a dictionary with keys: {list(target_info.keys())}")
        target_dict = target_info
    elif hasattr(target_info, 'to_dict'):  # pandas Series
        # [DEBUG] print(f"⚠️ target_info is a Series, converting")
        # If it's a Series, it's probably the binary target
        target_dict = {
            "binary_target": target_info,
            "type": DATASTORE.get("task_type", "zero_inflated_regression"),
            "zero_ratio": (target_info == 0).mean() if hasattr(target_info, 'mean') else 0
        }
    else:
        # [DEBUG] print(f"⚠️ Unknown target_info type, trying to get from DATASTORE")
        target_dict = DATASTORE.get("target_info", {})

    if not target_dict:
        # [DEBUG] print("❌ ERROR: No target information available")
        TRAINING_STATUS.update({
            "running": False,
            "done": True,
            "message": "Training failed: No target information found"
        })
        return {
            "type": "zero_inflated",
            "error": "No target information found",
            "training_time": time.time() - start_time
        }

    # [DEBUG] print(f"Target type: {target_dict.get('type', 'unknown')}")
    # [DEBUG] print(f"Zero ratio: {target_dict.get('zero_ratio', 0):.1%}")

    # Process form data correctly
    form_dict = {}
    for k, v in form_data.items():
        if isinstance(v, list):
            form_dict[k] = v[0] if len(v) == 1 else v
        else:
            form_dict[k] = v

    # Get modes from form
    model_mode = form_dict.get("model_mode", "auto")
    tuning_mode = form_dict.get("tuning_mode", "auto")

    # [DEBUG] print(f"Model mode: {model_mode}, Tuning mode: {tuning_mode}")

    # STAGE 1: Binary classification (zero vs non-zero)
    y_binary = None

    # Try multiple sources for binary target
    if "binary_target" in target_dict:
        y_binary = target_dict["binary_target"]
        # [DEBUG] print(f"✓ Got binary_target from target_dict: {len(y_binary)} samples")
    elif DATASTORE.get("y_binary") is not None:
        y_binary = DATASTORE["y_binary"]
        # [DEBUG] print(f"✓ Got y_binary from DATASTORE: {len(y_binary)} samples")
    elif DATASTORE.get("y") is not None:
        y_temp = DATASTORE["y"]
        # Check if it's binary (0/1)
        if hasattr(y_temp, 'unique'):
            unique_vals = y_temp.unique()
            if len(unique_vals) <= 2:
                y_binary = y_temp
                # [DEBUG] print(f"✓ Converted y to binary: {len(y_binary)} samples")

    if y_binary is None:
        # [DEBUG] print("❌ ERROR: Cannot find binary target")
        TRAINING_STATUS.update({
            "running": False,
            "done": True,
            "message": "Training failed: Cannot find binary target"
        })
        return {
            "type": "zero_inflated",
            "error": "No binary target found",
            "training_time": time.time() - start_time
        }

    # Align X and y
    idx = X.index.intersection(y_binary.index)
    X1 = X.loc[idx]
    y1 = y_binary.loc[idx]

    # [DEBUG] print(f"Stage 1: {len(y1)} samples, {len(y1[y1 == 1])} non-zero, {len(y1[y1 == 0])} zero")

    # Get classification models based on mode
    classification_models = _get_models_for_stage(
        form_dict, "classification",
        default_models=["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier", "knn_classifier"]
    )

    # [DEBUG] print(f"Stage 1 (Classification) models: {classification_models}")

    # Train stage 1
    stage1_results = train_models_with_manual_control(
        X1[selected_features],
        y1,
        "binary_classification",
        selected_features,
        {**form_dict, "models": classification_models}
    )

    # STAGE 2: Regression (non-zero values)
    stage2_results = {"error": "No regression target available"}
    nz_mask = y_binary == 1

    if nz_mask.sum() > 0:  # Check if we have non-zero samples
        X2 = X.loc[nz_mask]
        # Try to get regression target
        y2 = None

        # Try different possible regression targets
        if "regression_target" in target_dict:
            y2 = target_dict["regression_target"]
            # [DEBUG] print(f"✓ Got regression_target from target_dict: {len(y2)} samples")
        elif "regression_target_transformed" in target_dict:
            y2 = target_dict["regression_target_transformed"]
            # [DEBUG] print(f"✓ Got regression_target_transformed: {len(y2)} samples")
        elif DATASTORE.get("y_regression") is not None:
            y2 = DATASTORE["y_regression"]
            # [DEBUG] print(f"✓ Got y_regression from DATASTORE: {len(y2)} samples")
        elif DATASTORE.get("y_regression_transformed") is not None:
            y2 = DATASTORE["y_regression_transformed"]
            # [DEBUG] print(f"✓ Got y_regression_transformed: {len(y2)} samples")

        if y2 is not None and len(y2) > 0:
            idx2 = X2.index.intersection(y2.index)
            X2 = X2.loc[idx2]
            y2 = y2.loc[idx2]

            if len(X2) >= 20:  # Need enough samples
                # Get regression models based on mode
                regression_models = _get_models_for_stage(
                    form_dict, "regression",
                    default_models=["linear_regression", "ridge", "random_forest_regressor", "gradient_boosting_regressor"]
                )

                # CRITICAL FIX: Check if we have any regression models
                if not regression_models:
                    # [DEBUG] print(f"⚠️ No valid regression models found, using defaults")
                    regression_models = [
                        "linear_regression",
                        "ridge",
                        "random_forest_regressor",
                        "gradient_boosting_regressor"
                    ]

                # [DEBUG] print(f"Stage 2 (Regression) models: {regression_models}")
                # [DEBUG] print(f"Stage 2: {len(y2)} non-zero samples")

                # CRITICAL: Scale target if needed (apply to all Stage 2 models)
                # [DEBUG] print(f"Stage 2: Checking regression target scaling...")

                # Check if target needs scaling
                y2_mean = y2.mean()
                y2_std = y2.std()
                y2_max = y2.max()

                # [DEBUG] print(f"  -> Target stats: mean={y2_mean:.2e}, std={y2_std:.2e}, max={y2_max:.2e}")

                # Scale if target has extreme values
                if abs(y2_mean) > 1e6 or y2_std > 1e6 or abs(y2_max) > 1e6:
                    # [DEBUG] print(f"  -> Scaling Stage 2 target (mean={y2_mean:.2e}, std={y2_std:.2e})")

                    # Store original target for reference
                    y2_original = y2.copy()

                    from sklearn.preprocessing import RobustScaler
                    y2_scaler = RobustScaler()
                    y2_scaled = y2_scaler.fit_transform(y2.values.reshape(-1, 1)).flatten()
                    y2 = pd.Series(y2_scaled, index=y2.index)

                    # Store scaler for ALL Stage 2 models
                    if 'target_scalers' not in DATASTORE:
                        DATASTORE['target_scalers'] = {}

                    # Apply same scaler to all regression models in Stage 2
                    for reg_model in regression_models:
                        DATASTORE['target_scalers'][reg_model] = {
                            'scaler': y2_scaler,
                            'original_stats': {
                                'mean': y2_mean,
                                'std': y2_std,
                                'max': y2_max,
                                'median': y2_original.median()
                            },
                            'scaled': True,
                            'original_target': y2_original.tolist()[:100]  # Store sample for debugging
                        }

                    # [DEBUG] print(f"  -> Scaled target stored for {len(regression_models)} regression models")
                else:
                    pass  # placeholder (debug print removed)
                    # [DEBUG] print(f"  -> Target within reasonable range, no scaling needed")

                # Train stage 2
                stage2_results = train_models_with_manual_control(
                    X2[selected_features],
                    y2,
                    "regression",
                    selected_features,
                    {**form_dict, "models": regression_models}
                )
            else:
                stage2_results = {"error": f"Not enough non-zero samples: {len(X2)}"}
    else:
        # [DEBUG] print(f"⚠️ No non-zero samples found for stage 2")
        stage2_results = {"error": "No non-zero samples available"}

    # Calculate composite score - FIXED VERSION
    stage1_best_score = 0
    stage1_best_model_name = "None"
    stage2_best_score = 0
    stage2_best_model_name = "None"

    if stage1_results and "best_model" in stage1_results and stage1_results["best_model"]:
        stage1_best_score = stage1_results["best_model"].get("metrics", {}).get("accuracy", 0)
        stage1_best_model_name = stage1_results["best_model"].get("name", "Unknown")

    if stage2_results and "best_model" in stage2_results and stage2_results["best_model"]:
        stage2_best_score = stage2_results["best_model"].get("metrics", {}).get("r2", 0)
        stage2_best_model_name = stage2_results["best_model"].get("name", "Unknown")

    # Get zero ratio
    zero_ratio = target_dict.get("zero_ratio", 0)

    # Calculate composite score: Weighted average of stage1 accuracy and stage2 R²
    composite_score = (stage1_best_score * zero_ratio) + (stage2_best_score * (1 - zero_ratio))

    # Handle edge cases
    if np.isnan(composite_score) or np.isinf(composite_score):
        composite_score = 0

    # [DEBUG] print(f"\nComposite Score Calculation:")
    # [DEBUG] print(f"  Stage 1 Accuracy: {stage1_best_score:.4f} ({stage1_best_model_name})")
    # [DEBUG] print(f"  Stage 2 R²: {stage2_best_score:.4f} ({stage2_best_model_name})")
    # [DEBUG] print(f"  Zero Ratio: {zero_ratio:.1%}")
    # [DEBUG] print(f"  Composite Score: {composite_score:.4f}")

    # Combine results
    results = {
        "type": "zero_inflated",
        "stage1_classification": stage1_results,
        "stage2_regression": stage2_results,
        "combined_metrics": {
            "zero_ratio": zero_ratio,
            "non_zero_samples": len(y_binary[y_binary == 1]),
            "stage1_best_score": float(stage1_best_score),
            "stage1_best_model": stage1_best_model_name,
            "stage2_best_score": float(stage2_best_score),
            "stage2_best_model": stage2_best_model_name,
            "composite_score": float(composite_score)
        },
        "training_time": time.time() - start_time
    }

    # Update training status
    TRAINING_STATUS.update({
        "running": False,
        "done": True,
        "message": f"Zero-inflated training completed in {results['training_time']:.2f}s. Composite score: {composite_score:.3f}"
    })

    # [DEBUG] print(f"\n{'='*80}")
    # [DEBUG] print("ZERO-INFLATED TRAINING COMPLETE")
    # [DEBUG] print(f"Total time: {results['training_time']:.2f}s")
    # [DEBUG] print(f"Zero ratio: {zero_ratio:.1%}")
    # [DEBUG] print(f"Stage 1 best: {stage1_best_model_name} ({stage1_best_score:.4f})")
    # [DEBUG] print(f"Stage 2 best: {stage2_best_model_name} ({stage2_best_score:.4f})")
    # [DEBUG] print(f"Composite score: {composite_score:.3f}")
    # [DEBUG] print(f"{'='*80}")

    # Store in DATASTORE
    DATASTORE["zero_inflated_results"] = results

    return results

def ensure_zero_inflated_consistency():
    """Ensure zero-inflated data is properly aligned and stored"""
    task_type = DATASTORE.get("task_type")

    if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
        # [DEBUG] print(f"\n🔧 Ensuring zero-inflated data consistency...")

        # Get all components
        target_info = DATASTORE.get("target_info", {})
        X_original = DATASTORE.get("X_original")

        if X_original is not None and "binary_target" in target_info:
            # Align X with binary target
            y_binary = target_info["binary_target"]
            common_idx = X_original.index.intersection(y_binary.index)
            X_aligned = X_original.loc[common_idx]
            y_binary_aligned = y_binary.loc[common_idx]

            # Store aligned data
            DATASTORE["X_original"] = X_aligned
            DATASTORE["y_binary"] = y_binary_aligned
            DATASTORE["y"] = y_binary_aligned  # For FE compatibility

            # [DEBUG] print(f"  -> Aligned X: {X_aligned.shape}, y_binary: {y_binary_aligned.shape}")

            # Store regression target if available
            if "regression_target" in target_info:
                y_regression = target_info["regression_target"]
                y_regression_aligned = y_regression.loc[common_idx] if len(y_regression) > 0 else pd.Series()
                DATASTORE["y_regression"] = y_regression_aligned
                # [DEBUG] print(f"  -> Regression target: {y_regression_aligned.shape}")

            return True

    return False

# ============================================================================
# FIXED ZERO‑INFLATED DISPLAY FORMATTER
# ============================================================================
def format_zero_inflated_for_display(results):
    """
    Format zero-inflated results for template display with FIXED composition score
    """
    if not results or results.get("type") != "zero_inflated":
        return None

    formatted = {
        "stage1": {
            "title": "Stage 1: Zero Detection (Classification)",
            "models": [],
            "best_model": None,
            "metric_name": "Accuracy",
            "metric_value": 0
        },
        "stage2": {
            "title": "Stage 2: Value Prediction (Regression)",
            "models": [],
            "best_model": None,
            "metric_name": "R² Score",
            "metric_value": 0
        },
        "summary": {
            "zero_ratio": "0%",
            "composite_score": "0.0",
            "non_zero_samples": 0
        }
    }

    # Stage 1: Classification
    stage1 = results.get("stage1_classification", {})
    if stage1 and "test_metrics" in stage1:
        for model_key, model_info in stage1["test_metrics"].items():
            if isinstance(model_info, dict):
                formatted["stage1"]["models"].append({
                    "name": model_info.get("label", model_key),
                    "accuracy": f"{model_info.get('metrics', {}).get('accuracy', 0):.4f}",
                    "time": f"{model_info.get('training_time', 0):.2f}s"
                })

        # Get best model and its accuracy
        if stage1.get("best_model"):
            formatted["stage1"]["best_model"] = stage1["best_model"].get("name", "Unknown")
            formatted["stage1"]["metric_value"] = stage1["best_model"].get("metrics", {}).get("accuracy", 0)

    # Stage 2: Regression
    stage2 = results.get("stage2_regression", {})
    if stage2 and "test_metrics" in stage2:
        for model_key, model_info in stage2["test_metrics"].items():
            if isinstance(model_info, dict):
                # FIX: Handle NaN/Inf values for RMSE
                rmse_value = model_info.get('metrics', {}).get('rmse', 0)
                if rmse_value == 0 or np.isnan(rmse_value) or np.isinf(rmse_value):
                    rmse_display = "N/A"
                else:
                    rmse_display = f"{rmse_value:.6f}"

                formatted["stage2"]["models"].append({
                    "name": model_info.get("label", model_key),
                    "r2": f"{model_info.get('metrics', {}).get('r2', 0):.4f}",
                    "rmse": rmse_display,
                    "time": f"{model_info.get('training_time', 0):.2f}s"
                })

        # Get best model and its R² score
        if stage2.get("best_model"):
            formatted["stage2"]["best_model"] = stage2["best_model"].get("name", "Unknown")
            formatted["stage2"]["metric_value"] = stage2["best_model"].get("metrics", {}).get("r2", 0)

    # Combined metrics - FIXED COMPOSITE SCORE CALCULATION
    combined = results.get("combined_metrics", {})
    if combined:
        zero_ratio = combined.get("zero_ratio", 0)
        formatted["summary"]["zero_ratio"] = f"{zero_ratio:.1%}"
        formatted["summary"]["non_zero_samples"] = combined.get("non_zero_samples", 0)

        # FIX: Calculate composite score correctly
        # Weighted average: binary accuracy * zero ratio + R2 * (1 - zero ratio)
        stage1_accuracy = formatted["stage1"]["metric_value"]
        stage2_r2 = formatted["stage2"]["metric_value"]

        # Ensure values are valid
        if stage1_accuracy is None or np.isnan(stage1_accuracy):
            stage1_accuracy = 0
        if stage2_r2 is None or np.isnan(stage2_r2):
            stage2_r2 = 0

        # Calculate composite score
        composite_score = (stage1_accuracy * zero_ratio) + (stage2_r2 * (1 - zero_ratio))
        formatted["summary"]["composite_score"] = f"{composite_score:.3f}"

    # Sort models by performance
    formatted["stage1"]["models"].sort(key=lambda x: float(x["accuracy"]), reverse=True)
    formatted["stage2"]["models"].sort(key=lambda x: float(x["r2"]), reverse=True)

    return formatted

def format_rmse_for_display(rmse_value):
    """
    Format RMSE value for human-readable display
    """
    if rmse_value is None or np.isnan(rmse_value) or np.isinf(rmse_value):
        return "N/A"

    rmse = float(rmse_value)

    if rmse == 0:
        return "0.0"
    elif rmse < 0.001:
        return f"{rmse:.6f}"
    elif rmse < 1:
        return f"{rmse:.4f}"
    elif rmse < 1000:
        return f"{rmse:.2f}"
    elif rmse < 1_000_000:
        return f"{rmse/1000:.1f}K"
    elif rmse < 1_000_000_000:
        return f"{rmse/1_000_000:.1f}M"
    else:
        return f"{rmse/1_000_000_000:.1f}B"

def format_zero_inflated_results(results):
    """
    Format zero-inflated results for template display (legacy wrapper)
    """
    return format_zero_inflated_for_display(results)

def cleanup_resources():
    """Clean up joblib/loky resources to prevent memory leaks"""
    try:
        # Try newer joblib API first
        try:
            from loky import get_reusable_executor
            executor = get_reusable_executor()
            executor.shutdown(wait=True)
            # [DEBUG] print("  -> Cleaned up loky resources")
        except ImportError:
            # Fallback for older joblib versions
            try:
                from joblib.externals.loky import get_reusable_executor
                executor = get_reusable_executor()
                executor.shutdown(wait=True)
                # [DEBUG] print("  -> Cleaned up joblib resources")
            except ImportError:
                pass  # placeholder (debug print removed)
                # If loky is not available, just pass
                # [DEBUG] print("  -> Loky not available, skipping cleanup")
    except Exception as e:
        pass  # placeholder (debug print removed)
        # [DEBUG] print(f"  -> Resource cleanup failed: {str(e)}")

def inverse_transform_predictions(y_pred, model_key):
    """
    Inverse transform predictions to original scale using stored scalers
    """
    if model_key and 'target_scalers' in DATASTORE and model_key in DATASTORE['target_scalers']:
        scaler_info = DATASTORE['target_scalers'][model_key]
        if 'scaler' in scaler_info and scaler_info.get('scaled', False):
            try:
                scaler = scaler_info['scaler']
                y_pred_2d = np.array(y_pred).reshape(-1, 1)
                y_pred_orig = scaler.inverse_transform(y_pred_2d).flatten()
                # [DEBUG] print(f"    -> Inverse transformed predictions for {model_key}")
                return y_pred_orig
            except Exception as e:
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"    -> Could not inverse transform predictions: {e}")
    return y_pred

# ============================================================================
# FIXED MAIN TRAINING FUNCTION WITH DYNAMIC CV AND PRIMARY METRIC
# ============================================================================
def train_models_with_manual_control(X, y, task_type, selected_features, form_data, start_index=0):
    """
    Enhanced training with proper return structure for both regular and stage-based training.
    start_index : int – global starting model index (used for zero‑inflated stages).
    """
    global TRAINING_STATUS, CURRENT_FORM_DATA

    from sklearn.model_selection import cross_val_score

    # [DEBUG] print(f"\n{'='*80}")
    # [DEBUG] print(f"TRAIN_MODELS_WITH_MANUAL_CONTROL: Starting")
    # [DEBUG] print(f"Task type: {task_type}")
    # [DEBUG] print(f"Input y type: {type(y)}")
    # [DEBUG] print(f"Selected features: {len(selected_features)}")
    # [DEBUG] print(f"Start index: {start_index}")

    LINEAR_MODELS = {
        "linear_regression",
        "ridge",
        "lasso",
        "logistic_regression"
    }

    # ======================= HELPERS =======================
    def decide_hyperparameter_strategy(model_key, model_mode, tuning_mode):
        if model_mode == "manual" and tuning_mode == "manual":
            return "manual"
        if model_key in LINEAR_MODELS:
            return "grid"
        return "random"

    results = {
        "test_metrics": {},
        "train_metrics": {},
        "cv_scores": {},
        "best_model": None,
        "fit_analysis": {},
        "plot_paths": {},
        "used_hyperparams": {},
        "feature_importance": [],
        "training_time": None,
        "sampling_applied": {},
        "warnings": [],
        "type": "regular"
    }

    # ================= DUPLICATE FEATURE DETECTION =================
    def check_and_fix_identical_features(X_data):
        """
        Check for identical columns and remove duplicates to prevent R² = 1.0
        """
        X_clean = X_data.copy()
        removed_cols = []
        columns_to_check = list(X_data.columns)

        for i, col1 in enumerate(columns_to_check):
            if col1 not in X_clean.columns:
                continue

            for col2 in columns_to_check[i+1:]:
                if col2 not in X_clean.columns:
                    continue

                # Check if columns are identical
                try:
                    # Use correlation as a check
                    corr = X_clean[col1].corr(X_clean[col2])
                    if abs(corr) > 0.999 or X_clean[col1].equals(X_clean[col2]):
                        # [DEBUG] print(f"  ⚠️ Removing duplicate column: {col2} (identical/correlated with {col1})")
                        X_clean = X_clean.drop(columns=[col2])
                        removed_cols.append(col2)
                except:
                    # Try direct comparison
                    try:
                        if (X_clean[col1] == X_clean[col2]).all():
                            # [DEBUG] print(f"  ⚠️ Removing duplicate column: {col2} (identical to {col1})")
                            X_clean = X_clean.drop(columns=[col2])
                            removed_cols.append(col2)
                    except:
                        pass

        return X_clean, removed_cols

    def check_and_fix_collinearity(X_data):
        """Remove perfectly collinear features"""
        import numpy as np

        X_clean = X_data.copy()
        removed_cols = []

        if len(X_clean.columns) > 1:
            # Calculate correlation matrix
            corr_matrix = X_clean.corr().abs()

            # Create a set to track columns to remove
            columns_to_remove = set()

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.99:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        if col1 not in columns_to_remove:
                            # [DEBUG] print(f"  ⚠️ Removing collinear column: {col2} (corr={corr_matrix.iloc[i, j]:.3f} with {col1})")
                            columns_to_remove.add(col2)
                            removed_cols.append(col2)

            # Remove identified columns
            if columns_to_remove:
                X_clean = X_clean.drop(columns=list(columns_to_remove))

        return X_clean, removed_cols

    def validate_training_data(X_train, y_train, X_test, y_test):
        """Validate data to prevent unrealistic R² scores"""
        import numpy as np

        warnings = []

        # Check for constant target
        if y_train.nunique() == 1 or y_test.nunique() == 1:
            warnings.append("Target has only one unique value")

        # Check for very high correlation between features
        if len(X_train.columns) > 1:
            corr_matrix = X_train.corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append(f"{col1}-{col2}: {corr_matrix.iloc[i, j]:.3f}")

            if high_corr_pairs:
                warnings.append(f"Highly correlated features: {', '.join(high_corr_pairs[:3])}")

        # Check feature variance
        low_var_features = []
        for col in X_train.columns:
            if X_train[col].var() < 1e-10:
                low_var_features.append(col)

        if low_var_features:
            warnings.append(f"Low variance features: {', '.join(low_var_features[:3])}")

        return warnings

    def detect_data_leakage(X_train, y_train, X_test, y_test):
        """Detect if there's data leakage causing R² = 1.0 – warn only, don't abort."""
        warnings = []

        # Check if train and test are identical
        if X_train.shape == X_test.shape:
            if X_train.equals(X_test) or y_train.equals(y_test):
                warnings.append("CRITICAL: Train and test sets are identical!")

        # Check for overlapping indices
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        overlap = train_indices.intersection(test_indices)
        if overlap:
            warnings.append(f"CRITICAL: {len(overlap)} overlapping indices between train and test!")

        # Check for constant features
        for col in X_train.columns:
            if X_train[col].nunique() == 1:
                warnings.append(f"Constant feature in training: {col}")
            if X_test[col].nunique() == 1:
                warnings.append(f"Constant feature in testing: {col}")

        # Check if target is perfectly predictable from a single feature – warn only
        for col in X_train.columns:
            try:
                corr = X_train[col].corr(y_train)
                if abs(corr) > 0.9999:   # increased threshold and only warn
                    warnings.append(f"Feature '{col}' has near‑perfect correlation with target (r={corr:.4f})")
            except:
                pass

        return warnings

    # ================= DETECT IF THIS IS A STAGE TRAINING =================
    is_stage_training = False
    stage_info = {}
    stage_prefix = ""

    if isinstance(form_data, dict):
        stage = form_data.get("stage", "")
        stage_number = form_data.get("stage_number", "")
        stage_name = form_data.get("stage_name", "")

        if stage or stage_number:
            is_stage_training = True
            stage_info = {
                "stage": stage,
                "stage_number": stage_number,
                "stage_name": stage_name
            }
            stage_prefix = f"Stage {stage_number}: "
            # [DEBUG] print(f"⚠️  STAGE TRAINING DETECTED: {stage_name} (Stage {stage_number})")

            # Set training status message with stage info
            TRAINING_STATUS.update({
                "message": f"{stage_prefix}{stage_name}"
            })

    MIN_MODEL_DISPLAY_TIME = 0.5  # Reduced for faster UI

    # ================= RESET (only if not a stage) =================
    # For stage training, total_models is already set by the calling function.
    if not is_stage_training:
        TRAINING_STATUS.update({
            "running": True,
            "done": False,
            "progress": 0,
            "current_model": "Initializing...",
            "model_index": start_index,
            "total_models": 0,          # will be set after model selection
            "models_completed": 0,
            "models_remaining": 0,
            "stage": None,
            "stage_number": None,
            "message": "Preparing models..."
        })

    start_time = time.time()

    # ================= TASK =================
    task_group = "regression" if task_type == "regression" else "classification"
    registry = MODEL_REGISTRY[task_group]

    # Use different scoring metrics
    if task_group == "regression":
        scoring = "r2"
        best_score = float('-inf')
        primary_metric = "r2"
    else:
        scoring = "accuracy"
        best_score = 0
        primary_metric = "accuracy"

    # ================= GET FORM DATA =================
    def debug_form_parameters(form_data):
        """Debug form parameters"""
        # [DEBUG] print("\n" + "="*80)
        # [DEBUG] print("DEBUG FORM PARAMETERS:")

        form_dict = {}
        for key, value in form_data.items():
            if isinstance(value, list) and len(value) == 1:
                form_dict[key] = value[0]
                # [DEBUG] print(f"  {key}: {value[0]} (type: {type(value[0])})")
            elif isinstance(value, list):
                form_dict[key] = value
                # [DEBUG] print(f"  {key}: {value} (list)")
            else:
                form_dict[key] = value
                # [DEBUG] print(f"  {key}: {value} (type: {type(value)})")

        # [DEBUG] print(f"Manual parameters found:")
        for key in form_dict.keys():
            if key in ['alpha', 'n_estimators', 'max_depth', 'min_samples_leaf', 'n_neighbors', 'C', 'epsilon', 'kernel']:
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"  - {key}: {form_dict[key]}")

        # [DEBUG] print("="*80)
        return form_dict

    form_dict = debug_form_parameters(form_data)

    model_mode = form_dict.get("model_mode", "auto")
    # [DEBUG] print(f"DEBUG: model_mode after processing: {model_mode}")

    # CRITICAL FIX: If model_mode is auto, force tuning_mode to auto
    if model_mode == "auto":
        tuning_mode = "auto"
        # [DEBUG] print("DEBUG: Auto mode detected, setting tuning_mode = 'auto'")
    else:
        tuning_mode = form_dict.get("tuning_mode", "manual")
        # [DEBUG] print(f"DEBUG: Manual mode, tuning_mode: {tuning_mode}")

    # Handle both single and multiple model selection
    selected_models = []
    if "models" in form_dict:
        models_value = form_dict["models"]
        if isinstance(models_value, list):
            selected_models = models_value
        elif isinstance(models_value, str):
            selected_models = [models_value]
    # [DEBUG] print(f"DEBUG: Selected models: {selected_models}")

    # ================= DATASET INFO =================
    n_samples = len(X)
    n_features = len(selected_features)

    # [DEBUG] print(f"Dataset info: {n_samples:,} rows, {n_features} features")
    # [DEBUG] print(f"Model mode: {model_mode}, Tuning mode: {tuning_mode}")

    # ================= OPTIMIZED SAMPLING FOR LARGE DATASETS =================
    def optimized_sample_for_large_datasets(model_key, X_data, y_data, task_group, model_mode, tuning_mode, original_sample_count):
        """
        OPTIMIZED SAMPLING FOR LARGE DATASETS (UP TO LAKHS OF ROWS)
        - Scales based on dataset size
        - Maintains reasonable sample sizes for good performance
        - Handles all task types properly
        """
        n_samples = len(X_data)

        # [DEBUG] print(f"\n  === SAMPLING DEBUG FOR {model_key} ===")
        # [DEBUG] print(f"  Input X_data shape: {X_data.shape}")
        # [DEBUG] print(f"  Input y_data length: {len(y_data)}")
        # [DEBUG] print(f"  n_samples (current): {n_samples:,}")
        # [DEBUG] print(f"  original_sample_count: {original_sample_count:,}")
        # [DEBUG] print(f"  model_mode: {model_mode}, tuning_mode: {tuning_mode}")
        # [DEBUG] print(f"  task_group: {task_group}")

        # DETERMINE MAX SAMPLES BASED ON MODEL TYPE
        max_samples = None

        # LOGISTIC REGRESSION - USE SMALLER SAMPLES
        if model_key == "logistic_regression":
            if model_mode == "auto":
                max_samples = 40000  # Reduced from 80000
            elif model_mode == "manual" and tuning_mode == "auto":
                max_samples = 20000  # Reduced from 100000
            else:  # manual + manual
                max_samples = 20000  # Reduced from 120000
            # [DEBUG] print(f"  -> Logistic regression max samples: {max_samples:,}")

        # OTHER LINEAR MODELS
        elif model_key in ["linear_regression", "ridge", "lasso"]:
            if model_mode == "auto":
                max_samples = 50000
            elif model_mode == "manual" and tuning_mode == "auto":
                max_samples = 70000
            else:  # manual + manual
                max_samples = 90000
            # [DEBUG] print(f"  -> {model_key} max samples: {max_samples:,}")

        # TREE-BASED MODELS
        elif model_key in ["random_forest_regressor", "random_forest_classifier"]:
            if model_mode == "auto":
                max_samples = 25000   # enough signal without killing speed
            elif model_mode == "manual" and tuning_mode == "auto":
                max_samples = 50000
            else:  # manual + manual
                max_samples = 80000
            # [DEBUG] print(f"  -> {model_key} max samples: {max_samples:,}")

        # GRADIENT BOOSTING
        elif model_key in ["gradient_boosting_regressor", "gradient_boosting_classifier"]:
            if model_mode == "auto":
                max_samples = 12000   # GBM is O(n*trees*depth), keep tight
            elif model_mode == "manual" and tuning_mode == "auto":
                max_samples = 25000
            else:  # manual + manual
                max_samples = 50000
            # [DEBUG] print(f"  -> {model_key} max samples: {max_samples:,}")

        # MEMORY-INTENSIVE MODELS (SVM, KNN)
        elif model_key in ["svr", "svm", "knn_regressor", "knn_classifier"]:
            if model_mode == "auto":
                max_samples = 10000
            elif model_mode == "manual" and tuning_mode == "auto":
                max_samples = 20000
            else:  # manual + manual
                max_samples = 40000
            # [DEBUG] print(f"  -> {model_key} max samples: {max_samples:,}")

        # DEFAULT (adaboost and others)
        else:
            if model_mode == "auto":
                max_samples = 15000
            elif model_mode == "manual" and tuning_mode == "auto":
                max_samples = 25000
            else:  # manual + manual
                max_samples = 50000
            # [DEBUG] print(f"  -> Default max samples: {max_samples:,}")

        # SCALE BASED ON ORIGINAL DATASET SIZE
        if original_sample_count > 1000000:
            max_samples = int(max_samples * 1.5)
            # [DEBUG] print(f"  -> Scaled for large dataset: {max_samples:,}")
        elif original_sample_count > 500000:
            max_samples = int(max_samples * 1.2)
            # [DEBUG] print(f"  -> Scaled for medium dataset: {max_samples:,}")

        # NO SAMPLING NEEDED IF ALREADY UNDER LIMIT
        if n_samples <= max_samples:
            # [DEBUG] print(f"  -> No sampling needed ({n_samples:,} <= {max_samples:,})")
            # [DEBUG] print(f"  === END SAMPLING DEBUG ===\n")
            return X_data, y_data, False

        # [DEBUG] print(f"  -> Sampling needed: {n_samples:,} > {max_samples:,}")

        # INTELLIGENT SAMPLING
        try:
            if task_group == "classification" and y_data.nunique() > 1:
                # Stratified sampling for classification
                from sklearn.model_selection import train_test_split
                X_sampled, _, y_sampled, _ = train_test_split(
                    X_data, y_data,
                    train_size=max_samples,
                    stratify=y_data,
                    random_state=42
                )
                # [DEBUG] print(f"  -> Applied stratified sampling")

            elif hasattr(y_data, 'unique') and len(np.unique(y_data)) <= 2:
                # Binary classification - ensure class balance
                from sklearn.model_selection import train_test_split
                X_sampled, _, y_sampled, _ = train_test_split(
                    X_data, y_data,
                    train_size=max_samples,
                    stratify=y_data,
                    random_state=42
                )
                # [DEBUG] print(f"  -> Applied binary stratified sampling")

            else:
                # Random sampling
                sample_indices = np.random.choice(n_samples, max_samples, replace=False)
                if isinstance(X_data, pd.DataFrame):
                    X_sampled = X_data.iloc[sample_indices]
                    y_sampled = y_data.iloc[sample_indices]
                else:
                    X_sampled = X_data[sample_indices]
                    y_sampled = y_data[sample_indices]
                # [DEBUG] print(f"  -> Applied random sampling")

            # [DEBUG] print(f"  -> Final sampled size: {len(X_sampled):,} rows")
            # [DEBUG] print(f"  === END SAMPLING DEBUG ===\n")
            return X_sampled, y_sampled, True

        except Exception as e:
            # [DEBUG] print(f"  -> Sampling failed: {e}")
            # [DEBUG] print(f"  -> Using original data")
            # [DEBUG] print(f"  === END SAMPLING DEBUG ===\n")
            return X_data, y_data, False

    # ================= OPTIMIZED HYPERPARAMETER SEARCH =================
    def optimized_hyperparameter_search_fixed(model_key, pipeline, X_tr, y_tr, param_grid, scoring, search_type):
        """
        FIXED hyperparameter search that works for all models and dataset sizes
        WITH COMPREHENSIVE DATA CLEANING
        ENHANCED with Ridge-specific fixes
        """
        import numpy as np

        n_samples = len(X_tr)

        # ================= CRITICAL: CLEAN DATA BEFORE ANYTHING =================
        # [DEBUG] print(f"  -> Cleaning data before hyperparameter search...")
        X_tr_clean, y_tr_clean = validate_and_clean_training_data(X_tr, y_tr, model_key)

        if model_key == "lasso":
            # [DEBUG] print(f"  -> SPECIAL: Lasso-specific parameter grid optimization")
            # ========== CRITICAL: FIX TARGET SCALING ==========
            if isinstance(y_tr_clean, pd.Series):
                # Check target magnitude - this is causing convergence issues
                y_mean = y_tr_clean.mean()
                y_std = y_tr_clean.std()
                y_abs_max = y_tr_clean.abs().max()

                # [DEBUG] print(f"    -> Target stats: mean={y_mean:.2e}, std={y_std:.2e}, max_abs={y_abs_max:.2e}")

                # If target has extreme values, scale it
                if y_abs_max > 1e6 or y_std > 1e6:
                    # [DEBUG] print(f"    -> WARNING: Target has extreme values, applying robust scaling")
                    # Use RobustScaler for the target
                    from sklearn.preprocessing import RobustScaler
                    y_scaler = RobustScaler()
                    y_tr_clean_2d = y_tr_clean.values.reshape(-1, 1)
                    y_tr_clean = pd.Series(y_scaler.fit_transform(y_tr_clean_2d).flatten(),
                                        index=y_tr_clean.index)
                    # [DEBUG] print(f"    -> Scaled target to reasonable range")

            # ========== FIX FEATURE SCALING ==========
            if isinstance(X_tr_clean, pd.DataFrame):
                numeric_cols = X_tr_clean.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in X_tr_clean.columns:
                        col_data = X_tr_clean[col]
                        col_max = col_data.abs().max()
                        col_std = col_data.std()

                        # Check for extreme values
                        if col_max > 1e6 or col_std > 1e6:
                            # [DEBUG] print(f"    -> Column '{col}' has extreme values (max={col_max:.2e}, std={col_std:.2e})")
                            # Apply Winsorizing (capping extreme values)
                            q99 = col_data.quantile(0.995)
                            q01 = col_data.quantile(0.005)
                            if np.isfinite(q99) and np.isfinite(q01):
                                X_tr_clean[col] = col_data.clip(q01, q99)
                                # [DEBUG] print(f"    -> Capped '{col}' to [{q01:.2e}, {q99:.2e}]")

            # Simplify the alpha parameter space
            for param_key, param_values in param_grid.items():
                if 'alpha' in param_key:
                    if isinstance(param_values, list) and len(param_values) > 5:
                        # Use logarithmic spaced alpha values
                        import numpy as np
                        alphas = np.logspace(-4, 2, 5)  # 0.0001 to 100
                        param_grid[param_key] = list(alphas)
                        # [DEBUG] print(f"    -> Simplified alpha grid to log-spaced values: {alphas}")

            for param_key, param_values in param_grid.items():
                if 'tol' in param_key:
                    # Use MUCH looser tolerance for Lasso
                    if isinstance(param_values, list):
                        # Replace tight tolerances with looser ones
                        new_tols = []
                        for tol_val in param_values:
                            if tol_val < 1e-3:  # If too tight
                                new_tols.append(1e-3)  # Minimum reasonable tol
                            else:
                                new_tols.append(tol_val)
                        param_grid[param_key] = list(set(new_tols))  # Remove duplicates
                        # [DEBUG] print(f"    -> Adjusted tolerance values: {param_grid[param_key]}")
                    else:
                        # Single value - ensure it's reasonable
                        if param_values < 1e-3:
                            param_grid[param_key] = 1e-3
                            # [DEBUG] print(f"    -> Increased tolerance from {param_values} to 1e-3")

            # ========== ADD SELECTION PARAMETER ==========
            # 'random' selection often converges better for large datasets
            selection_key = None
            for key in param_grid.keys():
                if 'selection' in key:
                    selection_key = key
                    break

            if selection_key:
                param_grid[selection_key] = ['random']  # Force random selection
                # [DEBUG] print(f"    -> Set selection='random' for better convergence")
            else:
                # Add selection parameter if missing
                param_grid['model__selection'] = ['random']
                # [DEBUG] print(f"    -> Added selection='random' parameter")

            # ========== REDUCE PARAMETER COMBINATIONS ==========
            total_combinations = 1
            for values in param_grid.values():
                if isinstance(values, list):
                    total_combinations *= len(values)

            if total_combinations > 50:
                # [DEBUG] print(f"    -> Too many parameter combinations ({total_combinations}), simplifying...")
                # Drastically reduce grid
                simplified_grid = {}
                for param, values in param_grid.items():
                    if isinstance(values, list) and len(values) > 3:
                        # Take only 3 values: min, middle, max
                        simplified_grid[param] = [values[0], values[len(values)//2], values[-1]]
                    else:
                        simplified_grid[param] = values
                param_grid = simplified_grid
                # [DEBUG] print(f"    -> Reduced to {len(param_grid)} parameters")

            # Force reasonable max_iter values
            max_iter_key = None
            for key in param_grid.keys():
                if 'max_iter' in key:
                    max_iter_key = key
                    break

            if max_iter_key:
                if isinstance(param_grid[max_iter_key], list):
                    # Ensure all max_iter values are >= 1000
                    param_grid[max_iter_key] = [max(1000, val) for val in param_grid[max_iter_key]]
                    max_iter_vals = param_grid[max_iter_key]
                    # Convert numpy ints to regular Python ints for display
                    display_vals = []
                    for val in max_iter_vals:
                        if hasattr(val, 'item'):  # numpy type
                            display_vals.append(int(val.item()))
                        else:
                            display_vals.append(int(val))
                    # [DEBUG] print(f"    -> Ensured min max_iter=1000: {display_vals}")

            # Set n_jobs=1 for Lasso (parallel processing can cause convergence issues)
            # [DEBUG] print(f"    -> Setting n_jobs=1 for Lasso stability")

        # ================= SPECIAL RIDGE-SPECIFIC CLEANING =================
        elif model_key == "ridge":
            # [DEBUG] print(f"  -> SPECIAL: Enhanced cleaning for Ridge regression")

            # Apply specialized Ridge validation and cleaning
            X_tr_clean, y_tr_clean = validate_data_for_ridge(X_tr_clean, y_tr_clean)

            # Additional Ridge-specific checks
            if isinstance(X_tr_clean, pd.DataFrame):
                numeric_cols = X_tr_clean.select_dtypes(include=[np.number]).columns

                # Check for perfect multicollinearity (determinant close to zero)
                if len(numeric_cols) > 1:
                    # Add tiny noise to break perfect collinearity
                    np.random.seed(42)
                    noise = np.random.normal(0, 1e-8, X_tr_clean[numeric_cols].shape)
                    X_tr_clean[numeric_cols] = X_tr_clean[numeric_cols] + noise
                    # [DEBUG] print(f"    -> Added minimal noise to break perfect collinearity")

                # Ensure no extreme values that could cause overflow
                for col in numeric_cols:
                    if col in X_tr_clean.columns:
                        col_data = X_tr_clean[col]
                        # Cap extreme values at 99.9th percentile
                        upper_limit = col_data.quantile(0.999)
                        lower_limit = col_data.quantile(0.001)
                        if np.isfinite(upper_limit) and np.isfinite(lower_limit):
                            X_tr_clean[col] = col_data.clip(lower_limit, upper_limit)

        # ================= GENERAL CLEANING CONTINUES =================
        # Check for remaining NaN/Inf
        if isinstance(X_tr_clean, pd.DataFrame):
            nan_count = X_tr_clean.isna().sum().sum()
            inf_count = np.isinf(X_tr_clean.select_dtypes(include=[np.number])).sum().sum()
            if nan_count > 0 or inf_count > 0:
                # [DEBUG] print(f"  -> WARNING: Still found {nan_count} NaN and {inf_count} Inf values after cleaning!")
                # Force fill NaN
                X_tr_clean = X_tr_clean.fillna(0)
                # Replace Inf with large finite values
                numeric_cols = X_tr_clean.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in X_tr_clean.columns:
                        X_tr_clean[col] = X_tr_clean[col].replace([np.inf, -np.inf], 1e9)

        # Clean y if needed
        if isinstance(y_tr_clean, pd.Series):
            if y_tr_clean.isna().any():
                y_tr_clean = y_tr_clean.fillna(y_tr_clean.median() if y_tr_clean.notna().any() else 0)
            if np.any(np.isinf(y_tr_clean)):
                finite_vals = y_tr_clean[np.isfinite(y_tr_clean)]
                if len(finite_vals) > 0:
                    median_val = finite_vals.median()
                    y_tr_clean = y_tr_clean.replace([np.inf, -np.inf], median_val)
                else:
                    y_tr_clean = y_tr_clean.replace([np.inf, -np.inf], 0)

        # [DEBUG] print(f"  -> After cleaning: X shape {X_tr_clean.shape}, y length {len(y_tr_clean)}")

        # ================= DETERMINE CV STRATEGY =================
        cv = get_cv_strategy(model_key, y_tr_clean)  # Use the dynamic CV function

        # [DEBUG] print(f"  -> Using {cv}-fold CV for {search_type} search")

        # ================= FAST PRECISION GRID FOR TREE MODELS IN AUTO MODE =================
        if model_key in ["random_forest_regressor", "random_forest_classifier",
                          "gradient_boosting_regressor", "gradient_boosting_classifier"] and model_mode == "auto":
            # [DEBUG] print(f"  -> FAST AUTO GRID: Using precision param grid for {model_key}")

            if model_key == "random_forest_regressor":
                fast_grid = {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [6, 10, None],
                    "model__min_samples_leaf": [4, 8],
                    "model__max_features": ["sqrt", "log2"],
                }
            elif model_key == "random_forest_classifier":
                fast_grid = {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [6, 10, None],
                    "model__min_samples_leaf": [4, 8],
                    "model__max_features": ["sqrt"],
                }
            elif model_key == "gradient_boosting_regressor":
                fast_grid = {
                    "model__n_estimators": [80, 150],
                    "model__max_depth": [4, 6],
                    "model__learning_rate": [0.1, 0.2],
                    "model__subsample": [0.8],
                    "model__min_samples_leaf": [6, 10],
                }
            else:  # gradient_boosting_classifier
                fast_grid = {
                    "model__n_estimators": [80, 150],
                    "model__max_depth": [4, 6],
                    "model__learning_rate": [0.1, 0.2],
                    "model__subsample": [0.8],
                    "model__min_samples_leaf": [4, 8],
                }

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=fast_grid,
                n_iter=6,        # 6 iters over a small grid — precise and fast
                cv=cv,
                scoring=scoring,
                n_jobs=-1,       # parallel — safe for tree models
                random_state=42,
                verbose=0,
                error_score=np.nan
            )
            # [DEBUG] print(f"  -> Fast grid search configured: 6 iters, cv={cv}, n_jobs=-1")
            return search, cv, "random", X_tr_clean, y_tr_clean

        # ================= ENHANCE PARAM GRID FOR RIDGE =================
        if model_key == "ridge":
            # [DEBUG] print(f"  -> Enhancing parameter grid for Ridge stability")

            # CRITICAL FIX: Remove problematic 'sag' and 'lsqr' solvers completely
            for param_key, param_values in param_grid.items():
                if 'solver' in param_key:
                    # Filter out problematic solvers that produce NaN
                    problematic_solvers = ['sag', 'saga', 'lsqr']  # Added lsqr which was causing NaN
                    if isinstance(param_values, list):
                        filtered_solvers = [s for s in param_values if s not in problematic_solvers]
                        if filtered_solvers:
                            param_grid[param_key] = filtered_solvers
                            # [DEBUG] print(f"    -> Filtered out problematic solvers: {problematic_solvers}")
                            # [DEBUG] print(f"    -> Remaining solvers: {filtered_solvers}")
                        else:
                            # Use only stable solvers for Ridge
                            stable_solvers = ['auto', 'cholesky', 'svd']  # Only these are guaranteed stable
                            param_grid[param_key] = stable_solvers
                            # [DEBUG] print(f"    -> Using stable solvers: {stable_solvers}")
                    else:
                        # Replace single solver value if problematic
                        if param_values in problematic_solvers:
                            param_grid[param_key] = 'cholesky'  # Most stable default
                            # [DEBUG] print(f"    -> Replaced problematic solver '{param_values}' with 'cholesky'")

            # Ensure alpha parameter has appropriate values
            for param_key, param_values in param_grid.items():
                if 'alpha' in param_key:
                    # Make sure alpha values are positive and cover a good range
                    if isinstance(param_values, list):
                        # Filter out non-positive values
                        positive_values = []
                        for v in param_values:
                            try:
                                if isinstance(v, (int, float)) and v > 0:
                                    positive_values.append(float(v))
                            except:
                                pass

                        if not positive_values:
                            # Default good alpha values for Ridge
                            positive_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

                        param_grid[param_key] = positive_values
                        # [DEBUG] print(f"    -> Alpha values set to: {positive_values}")

            # Add/ensure max_iter is high enough
            max_iter_key = None
            for key in param_grid.keys():
                if 'max_iter' in key:
                    max_iter_key = key
                    break

            if max_iter_key:
                if isinstance(param_grid[max_iter_key], list):
                    # Ensure all max_iter values are sufficiently high
                    min_max_iter = min(param_grid[max_iter_key])
                    if min_max_iter < 5000:
                        param_grid[max_iter_key] = [max(5000, val) for val in param_grid[max_iter_key]]
                        # [DEBUG] print(f"    -> Increased max_iter to at least 5000: {param_grid[max_iter_key]}")
            else:
                # Add max_iter parameter if missing
                for key in param_grid.keys():
                    if key.endswith('__max_iter'):
                        param_grid[key] = [5000, 10000]
                    elif 'max_iter' in key:
                        param_grid[key] = [5000, 10000]

        if search_type == "grid":
            # Simplify grid for large datasets
            simplified_grid = {}
            for param, values in param_grid.items():
                if isinstance(values, list) and len(values) > 8:
                    if len(values) > 15:
                        simplified_grid[param] = [
                            values[0],
                            values[len(values)//4],
                            values[len(values)//2],
                            values[3*len(values)//4],
                            values[-1]
                        ]
                    else:
                        simplified_grid[param] = values[:8]
                else:
                    simplified_grid[param] = values

            # Apply logistic regression fixes if needed
            if model_key == "logistic_regression":
                # [DEBUG] print(f"  -> Applying logistic regression-specific grid optimization")
                simplified_grid = fix_logistic_regression_param_grid(simplified_grid)

            # CRITICAL: For Ridge, use error_score='raise' to catch NaN issues early
            if model_key == "ridge":
                # [DEBUG] print(f"  -> Using robust GridSearchCV for Ridge with error_score='raise'")
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=simplified_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,  # Single job for stability with Ridge
                    verbose=0,
                    error_score='raise'  # Raise error instead of returning NaN
                )
            else:
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=simplified_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    verbose=0,
                    error_score=np.nan
                )

        elif search_type == "random":
            n_iter = 5

            if model_key == "logistic_regression":
                n_iter = min(n_iter, 10)
                # [DEBUG] print(f"  -> Reduced iterations to {n_iter} for logistic regression stability")
            elif model_key == "ridge":
                # Use more iterations for Ridge to find better alpha
                n_iter = min(n_iter + 5, 25)
                # [DEBUG] print(f"  -> Increased iterations to {n_iter} for Ridge to better explore alpha space")

            # CRITICAL: For Ridge, use error_score='raise' to catch NaN issues early
            if model_key == "ridge":
                # [DEBUG] print(f"  -> Using robust RandomizedSearchCV for Ridge with error_score='raise'")
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,  # Single job for stability
                    random_state=42,
                    verbose=0,
                    error_score='raise'  # Raise error instead of returning NaN
                )
            else:
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    random_state=42,
                    verbose=0
                )

        else:  # manual mode
            return pipeline, cv, search_type

        # ================= RIDGE-SPECIFIC SEARCH CONFIG =================
        if model_key == "ridge":
            # [DEBUG] print(f"  -> Configuring Ridge-specific search parameters")

            # IMPORTANT: Ensure we use only stable solvers
            try:
                # Update the pipeline in search estimator
                current_params = search.estimator.get_params()

                # Find and update solver parameter
                solver_param = None
                for param_name in current_params:
                    if 'solver' in param_name:
                        solver_param = param_name
                        break

                if solver_param:
                    # Force stable solver
                    search.estimator.set_params(**{solver_param: 'cholesky'})
                    # [DEBUG] print(f"    -> Force setting solver to 'cholesky' (most stable)")

                # Increase max_iter significantly
                max_iter_param = None
                for param_name in current_params:
                    if 'max_iter' in param_name:
                        max_iter_param = param_name
                        break

                if max_iter_param:
                    search.estimator.set_params(**{max_iter_param: 10000})
                    # [DEBUG] print(f"    -> Set max_iter to 10000 for Ridge convergence")

                # Set tolerance
                tol_param = None
                for param_name in current_params:
                    if 'tol' in param_name:
                        tol_param = param_name
                        break

                if tol_param:
                    search.estimator.set_params(**{tol_param: 1e-4})
                    # [DEBUG] print(f"    -> Set tolerance to 1e-4 for Ridge")

                # CRITICAL: Update param_grid to exclude problematic solvers
                if hasattr(search, 'param_grid'):
                    for param_key in search.param_grid:
                        if 'solver' in param_key:
                            if isinstance(search.param_grid[param_key], list):
                                # Remove all problematic solvers
                                problematic = ['sag', 'saga', 'lsqr']
                                search.param_grid[param_key] = [
                                    s for s in search.param_grid[param_key]
                                    if s not in problematic
                                ]
                                # If empty, add stable solvers
                                if not search.param_grid[param_key]:
                                    search.param_grid[param_key] = ['cholesky', 'svd']
                                # [DEBUG] print(f"    -> Updated solver grid to exclude problematic solvers")

            except Exception as e:
                pass  # placeholder (debug print removed)
                # [DEBUG] print(f"    -> Could not configure Ridge parameters: {e}")
                # Continue anyway, but warn

        # Use cleaned data
        return search, cv, search_type, X_tr_clean, y_tr_clean

    # ================= INTELLIGENT MODEL SELECTION =================
    if model_mode == "auto":
        # AUTO MODE: Select models based on dataset size
        model_keys = []

        if n_samples < 10000:
            # Small dataset - include all models
            for m in registry.keys():
                if task_group == "regression" and m == "logistic_regression":
                    continue
                if task_group == "classification" and m in ["linear_regression", "ridge", "lasso"]:
                    continue
                model_keys.append(m)

        elif n_samples < 100000:
            # Medium dataset - exclude very slow models
            for m in registry.keys():
                if task_group == "regression" and m == "logistic_regression":
                    continue
                if task_group == "classification" and m in ["linear_regression", "ridge", "lasso"]:
                    continue
                if m in ["svr", "svm"] and n_samples > 50000:
                    continue  # SVM can be slow for >50k
                model_keys.append(m)

        else:
            # Large dataset - only scalable models
            scalable_models = [
                "linear_regression", "ridge", "lasso",
                "random_forest_regressor", "random_forest_classifier",
                "gradient_boosting_regressor", "gradient_boosting_classifier"
            ]

            for m in scalable_models:
                if m in registry:
                    # Task compatibility check
                    if task_group == "regression" and m == "logistic_regression":
                        continue
                    if task_group == "classification" and m in ["linear_regression", "ridge", "lasso"]:
                        continue
                    model_keys.append(m)

        # [DEBUG] print(f"Auto mode selected models: {model_keys}")

    else:  # Manual mode
        # Use user-selected models
        selected_models_list = []
        models_value = form_dict.get("models")

        if isinstance(models_value, list):
            selected_models_list = models_value
        elif isinstance(models_value, str):
            selected_models_list = [models_value]

        # Filter models based on task type and availability
        model_keys = []
        for model in selected_models_list:
            if model in registry:
                # Additional filtering for task type
                if task_group == "regression" and model in ["svm", "logistic_regression"]:
                    # [DEBUG] print(f"Skipping {model}: Not suitable for regression")
                    continue
                elif task_group == "classification" and model in ["svr", "linear_regression", "ridge", "lasso"]:
                    # [DEBUG] print(f"Skipping {model}: Not suitable for classification")
                    continue
                model_keys.append(model)

        # [DEBUG] print(f"DEBUG: Final model_keys to train: {model_keys}")
        # [DEBUG] print(f"DEBUG: Registry has keys: {list(registry.keys())}")

        # If no valid models selected, use defaults based on dataset size
        if not model_keys:
            # [DEBUG] print("No valid models selected, using default models")
            if n_samples < 50000:
                # Small-medium dataset defaults
                if task_group == "regression":
                    model_keys = ["linear_regression", "random_forest_regressor", "gradient_boosting_regressor"]
                else:  # classification
                    model_keys = ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]
            else:
                # Large dataset defaults
                if task_group == "regression":
                    model_keys = ["ridge", "random_forest_regressor"]
                else:  # classification
                    model_keys = ["logistic_regression", "random_forest_classifier"]

    # Ensure we don't have duplicates
    model_keys = list(dict.fromkeys(model_keys))

    # ================= NOW UPDATE TRAINING STATUS WITH TOTAL MODELS =================
    # For a stage, we keep the existing total_models and only adjust the start index.
    # For regular training, we set total_models here.
    if not is_stage_training:
        TRAINING_STATUS.update({
            "total_models": len(model_keys),
            "models_remaining": len(model_keys)
        })
    # Always set the model_index to the starting point (the loop will use global_idx)
    TRAINING_STATUS["model_index"] = start_index

    # [DEBUG] print(f"\nWill train {len(model_keys)} models (starting at global index {start_index+1})")
    # [DEBUG] print(f"Dataset size: {n_samples:,} rows")

    # ================= RESULTS =================
    # For stage training, we need a simpler structure
    if is_stage_training:
        results["type"] = "stage"
        results["stage_info"] = stage_info
    else:
        results["type"] = "regular"

    # ================= DATA SPLIT WITH DUPLICATE CHECK =================
    X = X[selected_features].copy()

    # CRITICAL FIX: Check for identical columns BEFORE training
    # [DEBUG] print(f"\nChecking for identical/duplicate features...")
    X_clean, removed_duplicates = check_and_fix_identical_features(X)

     # NEW: Check for perfect collinearity
    # [DEBUG] print(f"\nChecking for perfect collinearity...")
    X_clean, removed_collinear = check_and_fix_collinearity(X_clean)

    if removed_duplicates or removed_collinear:
        # [DEBUG] print(f"  ⚠️ Removed {len(removed_duplicates)} duplicate and {len(removed_collinear)} collinear columns")
        X = X_clean
        selected_features = list(X.columns)
        # [DEBUG] print(f"  -> Updated to {len(selected_features)} unique features")


    # CRITICAL: Ensure target isn't accidentally in features
    target_col = DATASTORE.get("target_column")
    if target_col and target_col in X.columns:
        # [DEBUG] print(f"  ⚠️ Target column '{target_col}' found in features! Removing...")
        X = X.drop(columns=[target_col])
        selected_features = [f for f in selected_features if f != target_col]

    # Split data FIRST - keep test set FULL
    if task_group == "classification" and y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # [DEBUG] print(f"\nData split:")
    # [DEBUG] print(f"  - Train: {len(X_train):,} rows")
    # [DEBUG] print(f"  - Test:  {len(X_test):,} rows")

    # Validate data
    data_warnings = validate_training_data(X_train, y_train, X_test, y_test)
    if data_warnings:
        # [DEBUG] print(f"  ⚠️ Data warnings: {data_warnings}")
        results["warnings"].extend(data_warnings)

    # Check for data leakage – WARN ONLY, DO NOT ABORT
    leakage_warnings = detect_data_leakage(X_train, y_train, X_test, y_test)
    if leakage_warnings:
        # [DEBUG] print(f"\n🚨 DATA LEAKAGE DETECTED (warnings only):")
        for warning in leakage_warnings:
            pass  # placeholder (debug print removed)
            # [DEBUG] print(f"  ⚠️  {warning}")
        results["warnings"].extend(leakage_warnings)

    # ================= TRAIN EACH MODEL =================
    best_pipeline = None
    best_model_name = None
    best_model_key = None
    best_model_metrics = {}
    best_train_metrics = {}

    for idx, model_key in enumerate(model_keys, start=1):
        global_idx = start_index + idx   # global model number for UI
        cfg = registry[model_key]
        label = cfg["label"]
        model_cls = cfg["model"]
        param_cfg = cfg.get("params", {}).copy()

        if is_stage_training:
            stage_name = stage_info.get("stage_name", "")
            TRAINING_STATUS.update({
                "current_model": label,
                "model_index": global_idx,
                "models_completed": global_idx - 1,
                "models_remaining": TRAINING_STATUS.get("total_models", 0) - (global_idx - 1),
                "message": f"{stage_name}: Training {label} ({global_idx}/{TRAINING_STATUS.get('total_models', '?')})",
                "progress": int(((global_idx - 1) / TRAINING_STATUS.get("total_models", 1)) * 100)
            })
        else:
            TRAINING_STATUS.update({
                "current_model": label,
                "model_index": idx,
                "models_completed": idx - 1,
                "models_remaining": len(model_keys) - (idx - 1),
                "message": f"Training {label} ({idx}/{len(model_keys)})",
                "progress": int(((idx - 1) / len(model_keys)) * 100)
            })

        # DEBUG: Show what's being passed to sampling
        # [DEBUG] print(f"  DEBUG before sampling:")
        # [DEBUG] print(f"  - X_train shape: {X_train.shape}")
        # [DEBUG] print(f"  - y_train length: {len(y_train)}")
        # [DEBUG] print(f"  - model_key: {model_key}")
        # [DEBUG] print(f"  - task_group: {task_group}")
        # [DEBUG] print(f"  - model_mode: {model_mode}")
        # [DEBUG] print(f"  - tuning_mode: {tuning_mode}")
        # [DEBUG] print(f"  - n_samples (original): {n_samples}")

        # Apply OPTIMIZED model-specific sampling for large datasets
        X_tr, y_tr, sampling_applied = optimized_sample_for_large_datasets(
            model_key, X_train, y_train, task_group, model_mode, tuning_mode, n_samples
        )

        # Clean the data to prevent NaN/inf issues
        X_tr, y_tr = validate_and_clean_training_data(X_tr, y_tr, model_key)

        if sampling_applied:
            # [DEBUG] print(f"  -> Sampled training data: {len(X_train):,} → {len(X_tr):,} rows")
            if not is_stage_training:
                results["sampling_applied"][model_key] = {
                    "original": len(X_train),
                    "sampled": len(X_tr)
                }
        else:
            pass  # placeholder (debug print removed)
            # [DEBUG] print(f"  -> Using full training data: {len(X_tr):,} rows")

        # Build OPTIMIZED pipeline for this model
        pipeline = build_optimized_pipeline(model_cls, X_tr, model_key)

        # Determine hyperparameter strategy
        search_type = decide_hyperparameter_strategy(model_key, model_mode, tuning_mode)
        cv_mean, cv_std = 0, 0

        try:
            # ========== HYPERPARAMETER TUNING STRATEGIES ==========
            model_start_time = time.time()
            cv_mean, cv_std = 0, 0  # Initialize CV scores

            # [DEBUG] print(f"\n  DEBUG: Hyperparameter strategy: {search_type}")
            # [DEBUG] print(f"    model_mode: {model_mode}, tuning_mode: {tuning_mode}")
            # [DEBUG] print(f"    param_cfg available: {len(param_cfg) > 0}")

            # ================= UNIFIED HYPERPARAMETER STRATEGY =================
            if model_mode == "auto":
                # AUTO TUNING MODE - FORCE HYPERPARAMETER TUNING
                # [DEBUG] print(f"  -> AUTO MODE: Performing hyperparameter tuning")

                if param_cfg and len(param_cfg) > 0:
                    # [DEBUG] print(f"  -> Parameter config available: {len(param_cfg)} parameters")

                    # Build parameter grid
                    param_grid = build_param_grid(param_cfg)
                    # [DEBUG] print(f"  -> Built param grid with {len(param_grid)} parameters")

                    # Use OPTIMIZED hyperparameter search
                    search, cv_folds, search_type_used, X_tr_clean, y_tr_clean = optimized_hyperparameter_search_fixed(
                        model_key, pipeline, X_tr, y_tr, param_grid, scoring, search_type
                    )

                    # Fit the search with robust error handling
                    # [DEBUG] print(f"  -> Starting hyperparameter search ({search_type_used}) with {cv_folds}-fold CV...")
                    search_start = time.time()

                    # Use robust hyperparameter search
                    search_result, search_status = robust_hyperparameter_search(
                        search, X_tr_clean, y_tr_clean, model_key, param_grid
                    )

                    search_time = time.time() - search_start

                    if search_result and search_status.startswith("success"):
                        # [DEBUG] print(f"  -> Search completed in {search_time:.1f}s")

                        trained = search_result.best_estimator_
                        results["used_hyperparams"][model_key] = search_result.best_params_

                        # Get cross-validation score
                        cv_mean = search_result.best_score_
                        if hasattr(search_result, 'cv_results_') and 'std_test_score' in search_result.cv_results_:
                            cv_std = search_result.cv_results_['std_test_score'][search_result.best_index_]
                        else:
                            cv_std = 0

                        # [DEBUG] print(f"  -> Best params: {search_result.best_params_}")
                        # [DEBUG] print(f"  -> CV Score: {cv_mean:.4f} (±{cv_std:.4f})")

                        if search_status == "success_after_clean":
                            pass  # placeholder (debug print removed)
                            # [DEBUG] print(f"  -> Note: Data cleaning was applied before search")

                    else:
                        # [DEBUG] print(f"  -> Hyperparameter search failed after {search_time:.1f}s")
                        # [DEBUG] print(f"  -> Falling back to default parameters")

                        # Ensure data is clean before fitting default model
                        X_tr_clean = X_tr.copy()
                        if isinstance(X_tr_clean, pd.DataFrame):
                            # Clean numeric columns
                            numeric_cols = X_tr_clean.select_dtypes(include=[np.number]).columns
                            for col in numeric_cols:
                                if col in X_tr_clean.columns:
                                    median_val = X_tr_clean[col].median()
                                    X_tr_clean[col] = X_tr_clean[col].fillna(median_val)

                        trained = pipeline.fit(X_tr_clean, y_tr)
                        results["used_hyperparams"][model_key] = "Default (search failed)"

                        # Calculate CV scores even for failed search
                        try:
                            from sklearn.model_selection import cross_val_score
                            cv_scores = cross_val_score(
                                pipeline, X_tr_clean, y_tr,
                                cv=get_cv_strategy(model_key, y_tr),
                                scoring=scoring,
                                n_jobs=-1
                            )
                            cv_mean = cv_scores.mean()
                            cv_std = cv_scores.std()
                            # [DEBUG] print(f"  -> Default CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                        except Exception as e:
                            # [DEBUG] print(f"  -> Could not calculate CV scores: {e}")
                            cv_mean, cv_std = 0, 0

                else:
                    # No parameters to tune
                    # [DEBUG] print(f"  -> No hyperparameters configured for {model_key}")
                    trained = pipeline.fit(X_tr, y_tr)
                    results["used_hyperparams"][model_key] = "Default (no param config)"

                    # Calculate CV scores for default parameters
                    try:
                        from sklearn.model_selection import cross_val_score
                        cv_scores = cross_val_score(
                            pipeline, X_tr, y_tr,
                            cv=get_cv_strategy(model_key, y_tr),
                            scoring=scoring,
                            n_jobs=-1
                        )
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                        # [DEBUG] print(f"  -> CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                    except Exception as e:
                        # [DEBUG] print(f"  -> Could not calculate CV scores: {e}")
                        cv_mean, cv_std = 0, 0

            elif model_mode == "manual" and tuning_mode == "auto":
                # MANUAL MODE WITH AUTO TUNING
                # [DEBUG] print(f"  -> MANUAL MODE WITH AUTO TUNING")

                if param_cfg and len(param_cfg) > 0:
                    # [DEBUG] print(f"  -> Parameter config available: {len(param_cfg)} parameters")

                    # Build parameter grid
                    param_grid = build_param_grid(param_cfg)
                    # [DEBUG] print(f"  -> Built param grid with {len(param_grid)} parameters")

                    # Use OPTIMIZED hyperparameter search
                    search, cv_folds, search_type_used, X_tr_clean, y_tr_clean = optimized_hyperparameter_search_fixed(
                        model_key, pipeline, X_tr, y_tr, param_grid, scoring, search_type
                    )

                    # Fit the search with robust error handling
                    # [DEBUG] print(f"  -> Starting hyperparameter search ({search_type_used})...")
                    search_start = time.time()

                    # Use robust hyperparameter search
                    search_result, search_status = robust_hyperparameter_search(
                        search, X_tr_clean, y_tr_clean, model_key, param_grid
                    )

                    search_time = time.time() - search_start

                    if search_result and search_status.startswith("success"):
                        # [DEBUG] print(f"  -> Search completed in {search_time:.1f}s")

                        trained = search_result.best_estimator_
                        results["used_hyperparams"][model_key] = search_result.best_params_

                        # Get cross-validation score
                        cv_mean = search_result.best_score_
                        if hasattr(search_result, 'cv_results_') and 'std_test_score' in search_result.cv_results_:
                            cv_std = search_result.cv_results_['std_test_score'][search_result.best_index_]
                        # [DEBUG] print(f"  -> Best params: {search_result.best_params_}")
                        # [DEBUG] print(f"  -> CV Score: {cv_mean:.4f} (±{cv_std:.4f})")

                    else:
                        # [DEBUG] print(f"  -> Hyperparameter search failed: {search_status}")
                        # [DEBUG] print(f"  -> Falling back to default parameters")
                        trained = pipeline.fit(X_tr, y_tr)
                        results["used_hyperparams"][model_key] = "Default (search failed)"

                else:
                    # No parameters to tune
                    # [DEBUG] print(f"  -> No hyperparameters configured for {model_key}")
                    trained = pipeline.fit(X_tr, y_tr)
                    results["used_hyperparams"][model_key] = "Default (no param config)"

            else:  # model_mode == "manual" and tuning_mode == "manual"
                # MANUAL TUNING MODE
                # [DEBUG] print(f"  -> MANUAL TUNING MODE")

                # Get ALL parameters from form data
                manual_params = extract_manual_params(form_dict, model_key, param_cfg)
                # [DEBUG] print(f"  -> DEBUG: Form dict keys: {list(form_dict.keys())[:10]}...")
                # [DEBUG] print(f"  -> DEBUG: Looking for {model_key} parameters")

                # Debug: Show what we're looking for
                for param in param_cfg.keys():
                    form_key = f"{model_key}__{param}"
                    # [DEBUG] print(f"    Checking for {form_key}: {'FOUND' if form_key in form_dict else 'NOT FOUND'}")

                # Also try model-specific extraction
                if not manual_params:
                    manual_params = get_model_specific_params(model_key, form_dict)

                # [DEBUG] print(f"  -> Manual params extracted: {len(manual_params)} parameters")
                if manual_params:
                    pass  # placeholder (debug print removed)
                    # [DEBUG] print(f"  -> Parameters: {manual_params}")

                manual_params = normalize_null_params(manual_params)
                # Validate and fix parameters
                validated_params = validate_and_fix_hyperparameters(model_key, manual_params, param_cfg)

                # Set parameters if any were provided
                if validated_params:
                    # [DEBUG] print(f"  -> Setting validated parameters for {model_key}: {validated_params}")
                    try:
                        # Set only valid parameters
                        valid_params = {}
                        for param_name, param_value in validated_params.items():
                            try:
                                # Test if parameter is valid by setting it
                                pipeline.set_params(**{param_name: param_value})
                                valid_params[param_name] = param_value
                            except Exception as e:
                                pass  # placeholder (debug print removed)
                                # [DEBUG] print(f"  -> Skipping invalid parameter {param_name}: {e}")

                        if valid_params:
                            pipeline.set_params(**valid_params)
                            results["used_hyperparams"][model_key] = valid_params
                            # [DEBUG] print(f"  -> Successfully set {len(valid_params)} parameters")
                    except Exception as e:
                        # [DEBUG] print(f"  -> Error setting parameters: {e}")
                        results["used_hyperparams"][model_key] = "Default (parameter setting failed)"

                # Train with manual params
                trained = pipeline.fit(X_tr, y_tr)

                # Calculate CV scores for manual parameters
                try:
                    cv_scores = cross_val_score(
                        pipeline, X_tr, y_tr,
                        cv=get_cv_strategy(model_key, y_tr),
                        scoring=scoring,
                        n_jobs=-1
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    # [DEBUG] print(f"  -> CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                except Exception as e:
                    # [DEBUG] print(f"  -> Could not calculate CV scores: {e}")
                    cv_mean, cv_std = 0, 0

            # Store cross-validation results
            results["cv_scores"][model_key] = {
                "mean": float(cv_mean),
                "std": float(cv_std),
                "n_folds": get_cv_strategy(model_key, y_tr)
            }

            # ========== EVALUATE ON TEST SET ==========
            # [DEBUG] print(f"  -> Evaluating on test set...")

            # Clean test data
            X_test_clean, y_test_clean = validate_and_clean_training_data(X_test, y_test, model_key)

            # Make predictions
            try:
                y_pred = trained.predict(X_test_clean)
                y_train_pred = trained.predict(X_tr) if len(X_tr) <= 10000 else None

                # Calculate metrics based on task type
                test_metrics = {}
                train_metrics = {}

                if task_type == "regression":
                    # Use helper function to get consistent metrics
                    test_metrics = calculate_consistent_regression_metrics(y_test_clean, y_pred, model_key)

                    if y_train_pred is not None:
                        train_metrics = calculate_consistent_regression_metrics(y_tr, y_train_pred, model_key)

                    # Log the scaling status
                    if model_key in DATASTORE.get('target_scalers', {}):
                        scaler_info = DATASTORE['target_scalers'][model_key]
                        if scaler_info.get('scaled', False):
                            pass  # placeholder (debug print removed)
                            # [DEBUG] print(f"    -> Using original scale RMSE: {test_metrics['rmse']:.2f}")
                            # [DEBUG] print(f"    -> Scaled RMSE: {test_metrics.get('rmse_scaled', 0):.4f}")

                    # Save regression plots if reasonable size
                    if len(y_test_clean) <= 5000:
                        try:
                            # Use inverse transformed predictions for plotting
                            y_pred_for_plot = inverse_transform_predictions(y_pred, model_key)
                            y_test_for_plot = y_test_clean

                            # If target was scaled, inverse transform it too
                            if model_key in DATASTORE.get('target_scalers', {}):
                                scaler_info = DATASTORE['target_scalers'][model_key]
                                if 'scaler' in scaler_info:
                                    y_test_2d = np.array(y_test_clean).reshape(-1, 1)
                                    y_test_for_plot = scaler_info['scaler'].inverse_transform(y_test_2d).flatten()

                            plot_path = save_regression_plots(y_test_for_plot, y_pred_for_plot, f"{model_key}_test")
                            results["plot_paths"][model_key] = plot_path
                        except Exception as e:
                            pass  # placeholder (debug print removed)
                            # [DEBUG] print(f"  -> Could not save plot: {e}")

                else:  # classification
                    test_metrics = {
                        "accuracy": float(accuracy_score(y_test_clean, y_pred)),
                        "precision": float(precision_score(y_test_clean, y_pred, average='weighted', zero_division=0)),
                        "recall": float(recall_score(y_test_clean, y_pred, average='weighted', zero_division=0)),
                        "f1": float(f1_score(y_test_clean, y_pred, average='weighted', zero_division=0))
                    }

                    if y_train_pred is not None:
                        train_metrics = {
                            "accuracy": float(accuracy_score(y_tr, y_train_pred)),
                            "precision": float(precision_score(y_tr, y_train_pred, average='weighted', zero_division=0)),
                            "recall": float(recall_score(y_tr, y_train_pred, average='weighted', zero_division=0)),
                            "f1": float(f1_score(y_tr, y_train_pred, average='weighted', zero_division=0))
                        }

                # Check for unrealistic R² scores (data leakage or perfect collinearity)
                if task_type == "regression" and abs(test_metrics["r2"]) > 0.999:
                    # [DEBUG] print(f"  ⚠️ WARNING: Suspiciously high R² = {test_metrics['r2']:.4f}")
                    results["warnings"].append(f"{label}: R² = {test_metrics['r2']:.4f} (possible data leakage or perfect collinearity)")

                # Store results
                results["test_metrics"][model_key] = {
                    "label": label,
                    "metrics": test_metrics,
                    "training_time": time.time() - model_start_time,
                    "cv_mean": float(cv_mean),
                    "cv_std": float(cv_std)
                }

                if train_metrics:
                    results["train_metrics"][model_key] = train_metrics

                # [DEBUG] print(f"  -> Test {primary_metric}: {test_metrics.get(primary_metric, 0):.4f}")

                # Update best model
                current_score = test_metrics.get(primary_metric, 0)
                if (task_type == "regression" and current_score > best_score) or \
                   (task_type != "regression" and current_score > best_score):
                    best_score = current_score
                    best_model_key = model_key
                    best_model_name = label
                    best_pipeline = trained
                    best_model_metrics = test_metrics
                    best_train_metrics = train_metrics

                    # [DEBUG] print(f"  -> New best model! {primary_metric}: {best_score:.4f}")

            except Exception as e:
                error_msg = str(e)[:100]
                # [DEBUG] print(f"  ❌ Evaluation failed: {error_msg}")
                results["test_metrics"][model_key] = {
                    "label": label,
                    "error": error_msg,
                    "training_time": time.time() - model_start_time
                }
                results["warnings"].append(f"{label}: Evaluation failed - {error_msg}")

        except Exception as e:
            error_msg = str(e)[:100]
            # [DEBUG] print(f"  ❌ Training failed: {error_msg}")
            results["test_metrics"][model_key] = {
                "label": label,
                "error": error_msg,
                "training_time": time.time() - model_start_time
            }
            results["warnings"].append(f"{label}: Training failed - {error_msg}")

        # Enforce minimum display time for UI
        elapsed = time.time() - model_start_time
        if elapsed < MIN_MODEL_DISPLAY_TIME:
            time.sleep(MIN_MODEL_DISPLAY_TIME - elapsed)

    # ================= FINALIZE RESULTS =================
    results["training_time"] = time.time() - start_time

    if best_model_key:
        results["best_model"] = {
            "key": best_model_key,
            "name": best_model_name,
            "metrics": best_model_metrics,
            "pipeline": best_pipeline,
            "train_metrics": best_train_metrics
        }

        # Create feature importance if applicable
        try:
            if hasattr(best_pipeline, 'named_steps'):
                model_step = best_pipeline.named_steps.get('model')
                if hasattr(model_step, 'feature_importances_'):
                    # Get feature names from the preprocessor
                    preprocessor = best_pipeline.named_steps.get('preprocessor')
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out()
                        importances = model_step.feature_importances_

                        # Create feature importance data
                        for name, importance in zip(feature_names, importances):
                            results["feature_importance"].append({
                                "feature": str(name),
                                "importance": float(importance)
                            })

                        # Sort by importance
                        results["feature_importance"].sort(key=lambda x: x["importance"], reverse=True)

                        # Create plot
                        if len(results["feature_importance"]) > 0:
                            top_n = min(15, len(results["feature_importance"]))
                            top_features = [item["feature"] for item in results["feature_importance"][:top_n]]
                            top_scores = [item["importance"] for item in results["feature_importance"][:top_n]]

                            # Create visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            y_pos = np.arange(len(top_features))
                            ax.barh(y_pos, top_scores, color='steelblue')
                            ax.set_yticks(y_pos)

                            # Truncate long feature names
                            labels = []
                            for f in top_features:
                                if len(f) > 30:
                                    labels.append(f[:27] + "...")
                                else:
                                    labels.append(f)

                            ax.set_yticklabels(labels)
                            ax.invert_yaxis()
                            ax.set_xlabel("Importance Score")
                            ax.set_title(f"Top {top_n} Feature Importance Scores")
                            ax.grid(axis='x', alpha=0.3, linestyle='--')

                            plt.tight_layout()

                            # Convert to base64
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
                            plt.close(fig)
                            buf.seek(0)

                            results["feature_importance_img"] = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            pass  # placeholder (debug print removed)
            # [DEBUG] print(f"  -> Could not create feature importance: {e}")

    # Clean up results for template
    results = cleanup_results_for_template(results)

    # Update training status
    TRAINING_STATUS.update({
        "running": False,
        "done": True,
        "message": f"Training completed in {results['training_time']:.2f}s. Best model: {best_model_name or 'None'}",
        "current_model": None,
        "model_index": 0
    })

    # [DEBUG] print(f"\n{'='*80}")
    # [DEBUG] print(f"TRAINING COMPLETE")
    # [DEBUG] print(f"Total time: {results['training_time']:.2f}s")
    # [DEBUG] print(f"Best model: {best_model_name or 'None'} ({best_model_key or 'None'})")
    if best_model_metrics:
        pass  # placeholder (debug print removed)
        # [DEBUG] print(f"Best {primary_metric}: {best_score:.4f}")
    # [DEBUG] print(f"{'='*80}")

    # Store in DATASTORE
    DATASTORE["training_results"] = results
    DATASTORE["best_model"] = best_pipeline

    return results
  
def build_param_grid(param_config):
    """
    Converts slider-based param configs into RandomizedSearchCV grids
    FIXED: Proper handling of logistic regression solver-penalty combinations
    """
    grid = {}

    if not param_config:
        return grid

    for param_full, cfg in param_config.items():

        # Case 1: cfg is already a list (e.g., ['auto', 'svd', 'cholesky'])
        if isinstance(cfg, list):
            if cfg:  # Only add if non-empty
                grid[param_full] = cfg
            continue

        # Case 2: cfg is a dict with type information
        if isinstance(cfg, dict):
            p_type = cfg.get("type")

            # Handle different parameter types
            if p_type == "int":
                min_val = int(cfg.get("min", 1))
                max_val = int(cfg.get("max", 100))
                step = int(cfg.get("step", 1))

                # Create values
                if max_val - min_val > 20:
                    # Sample values
                    import numpy as np
                    num_samples = min(5, (max_val - min_val) // step)
                    values = list(np.linspace(min_val, max_val, num_samples, dtype=int))
                else:
                    values = list(range(min_val, max_val + 1, step))

                grid[param_full] = values

            elif p_type == "float":
                min_val = float(cfg.get("min", 0.01))
                max_val = float(cfg.get("max", 10.0))
                step = float(cfg.get("step", 0.1))

                # Create values
                if (max_val - min_val) / step > 15:
                    import numpy as np
                    num_samples = min(5, int((max_val - min_val) / step))
                    values = list(np.round(np.linspace(min_val, max_val, num_samples), 3))
                else:
                    values = []
                    v = min_val
                    while v <= max_val + 1e-9:
                        values.append(round(v, 3))
                        v += step

                grid[param_full] = values

            elif p_type == "categorical":
                # Get options from either "options" or "values" key
                options = cfg.get("options", cfg.get("values", []))
                if options:
                    # SPECIAL HANDLING FOR LOGISTIC REGRESSION SOLVER
                    if 'solver' in param_full and 'logistic_regression' in param_full:
                        # Filter out problematic solvers
                        safe_solvers = []
                        problematic_solvers = ['saga', 'sag']  # These can cause convergence issues

                        for solver in options:
                            if solver not in problematic_solvers:
                                safe_solvers.append(solver)

                        if safe_solvers:
                            grid[param_full] = safe_solvers
                        else:
                            # Default to lbfgs if all solvers are problematic
                            grid[param_full] = ['lbfgs']
                    else:
                        grid[param_full] = options
                else:
                    # If no options, use default if available
                    default_val = cfg.get("default", None)
                    if default_val is not None:
                        grid[param_full] = [default_val]

            elif p_type == "bool":
                grid[param_full] = [True, False]

            else:
                # Unknown type, try to use as-is
                if "options" in cfg:
                    grid[param_full] = cfg.get("options", [])
                elif "values" in cfg:
                    grid[param_full] = cfg.get("values", [])

        # Case 3: cfg is a single value (e.g., True, False, or a number)
        else:
            grid[param_full] = [cfg]

    # CRITICAL FIX: If this is for logistic regression, apply special fixes
    if any('logistic_regression' in key for key in grid.keys()):
        # [DEBUG] print("  -> Applying logistic regression parameter grid fixes")
        grid = fix_logistic_regression_param_grid(grid)

    # Final safety check: remove any empty sequences
    grid = {k: v for k, v in grid.items() if v and len(v) > 0}

    return grid

def get_column_analysis(df):
    """Generate analysis for each column to display in target selection"""
    column_analysis = []

    for col in df.columns:
        # Start with is_index as False (most columns aren't index columns)
        is_index_column = False

        # Only very specific cases should be marked as index
        col_lower = str(col).lower()

        # Check for pandas default index columns (very specific patterns)
        if (col_lower in ['unnamed: 0', 'unnamed_0', 'index'] or
            (col_lower.startswith('unnamed:') and 'index' in col_lower)):
            is_index_column = True
        # Check for sequential integer columns ONLY if it's exactly 0,1,2,... or 1,2,3,...
        elif df[col].dtype in ['int64', 'int32', 'int'] and len(df) > 10:
            try:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    # Check if it's perfectly sequential from 0 or 1
                    sorted_vals = non_null.sort_values().reset_index(drop=True)
                    if len(sorted_vals) > 5:
                        # Create expected sequence
                        first_val = int(sorted_vals.iloc[0])
                        expected = pd.Series(range(first_val, first_val + len(sorted_vals)))

                        # Check if matches perfectly
                        if sorted_vals.reset_index(drop=True).equals(expected):
                            is_index_column = True
            except:
                pass

        col_type = detect_column_type(df[col])
        unique_count = df[col].nunique()
        missing_pct = df[col].isna().mean() * 100

        # Check for zero-inflation for numeric columns - FIX: Use .all() or .any()
        is_zero_inflated = False
        zero_percentage = 0
        if col_type in ["numeric", "numeric_skewed", "numeric_discrete", "numeric_low_cardinality"]:
            # FIX: Convert to numeric first, then check zeros
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            zero_mask = numeric_series == 0
            # FIX: Use .any() to check if there are any zeros, not the series itself
            if zero_mask.any():
                zero_percentage = zero_mask.mean() * 100
                if zero_percentage > 30 and zero_percentage < 95:
                    is_zero_inflated = True

        # Don't suggest index columns
        suggested = False
        reason = ""

        # FIX: Use proper boolean comparisons
        if not is_index_column and missing_pct < 30:
            if is_zero_inflated:
                # Suggest for zero-inflated modeling
                non_zero = df[col][df[col] != 0]
                # FIX: Check length properly
                if len(non_zero) > 50 and non_zero.nunique() > 5:
                    suggested = True
                    reason = f"Good for zero-inflated modeling ({zero_percentage:.1f}% zeros)"

            elif col_type in ["numeric", "numeric_skewed", "numeric_discrete", "numeric_low_cardinality"] and unique_count > 10:
                suggested = True
                reason = "Good for regression (continuous numeric)"

            elif col_type == "binary" and unique_count == 2:
                suggested = True
                reason = "Good for binary classification"

            elif col_type == "categorical" and 2 <= unique_count <= 20:
                suggested = True
                reason = f"Good for classification ({unique_count} classes)"

            elif col_type == "ordinal" and 3 <= unique_count <= 10:
                suggested = True
                reason = "Good for ordinal classification/regression"

        column_analysis.append({
            "name": col,
            "type": col_type,
            "unique": unique_count,
            "missing": f"{missing_pct:.1f}%",
            "suggested": suggested,
            "reason": reason,
            "is_index": is_index_column,
            "is_zero_inflated": is_zero_inflated,
            "zero_percentage": f"{zero_percentage:.1f}%" if is_zero_inflated else "0%"
        })

    # Sort by suggested first, then by type
    column_analysis.sort(key=lambda x: (
        not x["suggested"],
        not x["is_zero_inflated"],
        x["type"] != "numeric",
        x["name"]
    ))

    return column_analysis

def clean_undefined_parameters(form_dict):
    """
    Remove parameters with undefined values and validate parameter types
    """
    cleaned = {}

    for key, value in form_dict.items():
        if key == "models":
            # Keep models list as is
            cleaned[key] = value
        elif "__" in key:  # Model parameter
            # Check if value is meaningful
            if value is None or value == "" or str(value).lower() == "undefined":
                continue

            # Type validation based on parameter name
            if key.endswith("__copy_X") or key.endswith("__fit_intercept") or key.endswith("__positive"):
                # Boolean parameters
                if isinstance(value, str):
                    cleaned[key] = value.lower() in ["true", "1", "yes", "y"]
                else:
                    cleaned[key] = bool(value)
            elif key.endswith("__max_iter") or key.endswith("__n_estimators") or key.endswith("__n_neighbors"):
                # Integer parameters
                try:
                    cleaned[key] = int(float(value))
                except:
                    cleaned[key] = 100  # Default
            elif key.endswith("__alpha") or key.endswith("__C") or key.endswith("__learning_rate"):
                # Float parameters
                try:
                    cleaned[key] = float(value)
                except:
                    cleaned[key] = 1.0  # Default
            elif key.endswith("__solver") or key.endswith("__kernel") or key.endswith("__penalty"):
                # String parameters
                cleaned[key] = str(value)
            else:
                # Generic parameter
                cleaned[key] = value
        else:
            # Non-parameter keys
            cleaned[key] = value

    # [DEBUG] print(f"Cleaned parameters: {len([k for k in cleaned if '__' in k])} parameters retained")
    return cleaned

def generate_comprehensive_report(training_results, feature_engineering_summary=None):
    """
    Generate comprehensive report for download
    """
    report = {
        "summary": {
            "training_time": training_results.get("training_time", 0),
            "best_model": training_results.get("best_model", {}).get("name", "Unknown"),
            "total_models": len(training_results.get("test_metrics", {})),
            "type": training_results.get("type", "regular"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "model_comparison": {},
        "feature_engineering": feature_engineering_summary or {},
        "warnings": training_results.get("warnings", []),
        "model_details": {}
    }

    # Add model comparison data
    for model_key, model_info in training_results.get("test_metrics", {}).items():
        if isinstance(model_info, dict):
            report["model_comparison"][model_key] = {
                "label": model_info.get("label", model_key),
                "metrics": model_info.get("metrics", {}),
                "training_time": model_info.get("training_time", 0),
                "cv_score": model_info.get("cv_mean", 0)
            }

    # Add best model details
    if training_results.get("best_model"):
        report["best_model_details"] = {
            "name": training_results["best_model"].get("name", "Unknown"),
            "metrics": training_results["best_model"].get("metrics", {}),
            "key": training_results["best_model"].get("key", "unknown")
        }

    # Add zero-inflated details if present
    if training_results.get("type") == "zero_inflated":
        report["zero_inflated_details"] = training_results.get("combined_metrics", {})

        # Fix composition score calculation
        stage1 = training_results.get("stage1_classification", {})
        stage2 = training_results.get("stage2_regression", {})

        # Calculate weighted composite score
        zero_ratio = training_results.get("combined_metrics", {}).get("zero_ratio", 0)

        stage1_score = 0
        if stage1 and stage1.get("best_model"):
            stage1_score = stage1["best_model"].get("metrics", {}).get("accuracy", 0)

        stage2_score = 0
        if stage2 and stage2.get("best_model"):
            stage2_score = stage2["best_model"].get("metrics", {}).get("r2", 0)

        # Weighted composite score: binary accuracy * zero ratio + R2 * (1 - zero ratio)
        composite_score = (stage1_score * zero_ratio) + (stage2_score * (1 - zero_ratio))
        report["zero_inflated_details"]["composite_score"] = composite_score

    return report

def get_model_display_name(model_key):
    """Get display name for model key"""
    model_names = {
        "linear_regression": "Linear Regression",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "logistic_regression": "Logistic Regression",
        "random_forest_regressor": "Random Forest Regressor",
        "random_forest_classifier": "Random Forest Classifier",
        "gradient_boosting_regressor": "Gradient Boosting Regressor",
        "gradient_boosting_classifier": "Gradient Boosting Classifier",
        "svr": "Support Vector Regressor",
        "svm": "Support Vector Machine",
        "knn_regressor": "K-Nearest Neighbors Regressor",
        "knn_classifier": "K-Nearest Neighbors Classifier",
        "adaboost_regressor": "AdaBoost Regressor",
        "adaboost_classifier": "AdaBoost Classifier"
    }
    return model_names.get(model_key, model_key)

def format_number_filter(value):
    """Jinja2 filter: format numbers with commas and smart decimal places."""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "N/A"
            if value == int(value):
                return f"{int(value):,}"
            if abs(value) >= 1:
                return f"{value:,.4f}"
            return f"{value:.6f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)
    except Exception:
        return str(value)

def format_model_comparison_for_display(results):
    """
    Format model comparison results for template display
    """
    if not results or "test_metrics" not in results:
        return []

    model_comparison = []

    for model_key, model_info in results["test_metrics"].items():
        if isinstance(model_info, dict):
            metrics = model_info.get("metrics", {})

            # Format metrics for display
            formatted_metrics = {}
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name in ["r2", "accuracy", "precision", "recall", "f1"]:
                        formatted_metrics[metric_name] = f"{metric_value:.4f}"
                    elif metric_name in ["rmse", "mae"]:
                        formatted_metrics[metric_name] = f"{metric_value:.6f}"
                    else:
                        formatted_metrics[metric_name] = f"{metric_value:.4f}"
                else:
                    formatted_metrics[metric_name] = str(metric_value)

            model_comparison.append({
                "name": model_info.get("label", model_key),
                "key": model_key,
                "metrics": formatted_metrics,
                "training_time": f"{model_info.get('training_time', 0):.2f}s",
                "cv_score": f"{model_info.get('cv_mean', 0):.4f}",
                "is_best": results.get("best_model", {}).get("key") == model_key
            })

    # Sort by primary metric
    if model_comparison:
        # Determine primary metric
        primary_metric = None
        if "r2" in model_comparison[0]["metrics"]:
            primary_metric = "r2"
        elif "accuracy" in model_comparison[0]["metrics"]:
            primary_metric = "accuracy"

        if primary_metric:
            model_comparison.sort(
                key=lambda x: float(x["metrics"].get(primary_metric, 0)),
                reverse=True
            )

    return model_comparison
import os
import math
import io
import base64
import json
import time
import logging
import threading
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import seaborn as sns

from datetime import datetime
from flask import Flask, render_template, request, redirect, send_file, url_for, jsonify, flash, session, Response
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from models_registry import MODEL_REGISTRY
from state import *
from utility import *

# ---------------------------------------------------------------------------
# Warnings suppression
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Objective did not converge')
warnings.filterwarnings('ignore', message='Duality gap')
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Logging (replaces all print statements)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-fallback-key-change-in-production')

# ---------------------------------------------------------------------------
# Jinja2 filters  (defined once, registered once)
# ---------------------------------------------------------------------------
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

def float_filter(value):
    """Convert to float safely."""
    try:
        return float(value)
    except Exception:
        return 0.0

def replace_filter(s, old, new):
    """Jinja2 replace filter."""
    return s.replace(old, new)

def title_filter(s):
    """Jinja2 title filter."""
    return s.title()

app.jinja_env.filters['format_number']        = format_number_filter
app.jinja_env.filters['format_number_filter'] = format_number_filter
app.jinja_env.filters['float']                = float_filter
app.jinja_env.filters['replace_str']          = replace_filter
app.jinja_env.filters['title_str']            = title_filter

# ---------------------------------------------------------------------------
# Context processors  (single consolidated processor)
# ---------------------------------------------------------------------------
@app.context_processor
def inject_template_context():
    """Expose all necessary variables and helpers to every template."""
    return {
        "DATASTORE":            DATASTORE,
        "TRAINING_STATUS":      TRAINING_STATUS,
        "datetime":             datetime,
        "now":                  datetime.now,
        "format_number_filter": format_number_filter,
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """Main upload page."""
    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return render_template("index.html", error="No file selected")

        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Only CSV files allowed")

        try:
            DATASTORE.clear()
            DATASTORE["processing_complete"] = False

            file.seek(0)
            content = file.read().decode('utf-8', errors='ignore')

            df = None
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep, engine='python')
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue

            if df is None or len(df.columns) <= 1:
                df = pd.read_csv(io.StringIO(content), engine='python')

            if df is None or len(df) == 0:
                return render_template("index.html", error="Empty dataset")

            if len(df.columns) == 0:
                return render_template("index.html", error="Dataset has no columns")

            DATASTORE["original_df"] = df
            DATASTORE["original_missing"] = {
                "numeric":     int(df.select_dtypes(include="number").isna().sum().sum()),
                "categorical": int(df.select_dtypes(include="object").isna().sum().sum())
            }

            df_clean = df.copy()
            df_clean.columns = [str(col).strip().lower().replace(' ', '_') for col in df_clean.columns]
            df_clean = df_clean.replace(["", " ", "NA", "N/A", "null", "None", "?", "nan"], np.nan)

            cols_to_remove = []
            for col in df_clean.columns:
                col_lower = str(col).lower()
                if (col_lower.startswith('unnamed') or
                        col_lower == 'index' or
                        col_lower == 'id' or
                        col_lower == 'row' or
                        ('unnamed' in col_lower and
                         col_lower.replace('unnamed', '').replace(':', '').replace('_', '').isdigit())):
                    cols_to_remove.append(col)

            if cols_to_remove:
                df_clean = df_clean.drop(columns=cols_to_remove)
                logger.info("Removed index columns during upload: %s", cols_to_remove)

            empty_cols = df_clean.columns[df_clean.isna().all()].tolist()
            if empty_cols:
                df_clean = df_clean.drop(columns=empty_cols)

            DATASTORE["cleaned_df"]          = df_clean
            DATASTORE["current_df"]          = df_clean.copy()
            DATASTORE["processing_complete"] = True
            DATASTORE["eda_report"]          = generate_eda_report(df_clean)

            return redirect(url_for("dataset_overview"))

        except Exception as e:
            error_msg = str(e)
            if "MemoryError" in error_msg:
                error_msg = "File too large. Try a smaller file or sample your data first."
            elif "utf-8" in error_msg.lower():
                error_msg = "Encoding error. Try saving the file as UTF-8 CSV."
            return render_template("index.html", error=f"Error processing file: {error_msg}")

    return render_template("index.html")


@app.route("/overview")
def dataset_overview():
    df = DATASTORE.get("cleaned_df")
    if df is None:
        return redirect(url_for("index"))

    eda_report = generate_eda_report(df)

    try:
        df_viz = df.sample(5000, random_state=42) if len(df) > 10000 else df
        viz_data = create_visualizations(df_viz)
    except Exception as e:
        logger.warning("Visualization error: %s", e)
        viz_data = {}

    sample_data = df.head(10).to_html(
        classes="table table-striped table-bordered",
        index=False,
        na_rep="N/A"
    )

    return render_template(
        "dataset_overview.html",
        overview=eda_report["overview"],
        columns=eda_report["columns"],
        sample_data=sample_data,
        viz_data=viz_data,
        column_count=len(df.columns),
        row_count=len(df)
    )


@app.route("/target", methods=["GET", "POST"])
def select_target():
    """Enhanced target selection with automatic zero-inflated detection."""
    df = DATASTORE.get("cleaned_df")
    if df is None:
        return redirect(url_for("index"))

    dataset_size    = len(df)
    is_large_dataset = dataset_size > 10000

    if "column_analysis" not in DATASTORE:
        DATASTORE["column_analysis"] = get_column_analysis(df)
    column_analysis = DATASTORE["column_analysis"]

    if request.method == "POST":
        target_col       = request.form.get("target")
        auto_clean       = request.form.get("auto_clean") == "on"
        feature_strategy = request.form.get("feature_strategy", "auto")

        if not target_col:
            return render_template(
                "target.html",
                columns=df.columns.tolist(),
                column_analysis=column_analysis,
                dataset_size=dataset_size,
                is_large_dataset=is_large_dataset,
                error="Please select a target column"
            )

        try:
            logger.info("TARGET SELECTION: Processing column '%s'", target_col)
            DATASTORE["target_column"] = target_col

            # ------------------------------------------------------------------
            # 1. Auto-clean
            # ------------------------------------------------------------------
            if auto_clean:
                logger.info("Applying auto-cleaning to features...")
                feature_cols = [col for col in df.columns if col != target_col]

                numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                    logger.info("  Imputed %d numeric columns with median", len(numeric_cols))

                cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
                for col in cat_cols:
                    if col in df.columns:
                        mode_val = df[col].mode()
                        df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'missing')
                logger.info("  Imputed %d categorical columns with mode", len(cat_cols))

                DATASTORE["cleaned_df"]        = df
                DATASTORE["current_df"]        = df
                DATASTORE["auto_clean_applied"] = True
            else:
                DATASTORE["auto_clean_applied"] = False

            # ------------------------------------------------------------------
            # 2. Prepare target variable
            # ------------------------------------------------------------------
            logger.info("Analyzing target column '%s'...", target_col)
            target_series = df[target_col]

            target_stats = {
                "total_rows":        len(target_series),
                "missing_count":     int(target_series.isna().sum()),
                "missing_percentage": float(target_series.isna().mean() * 100),
                "zero_count":        int((target_series == 0).sum() if is_numeric_dtype(target_series) else 0),
                "zero_percentage":   float((target_series == 0).mean() * 100 if is_numeric_dtype(target_series) else 0),
                "unique_values":     int(target_series.nunique()),
                "dtype":             str(target_series.dtype)
            }

            target_info = prepare_target(target_series)
            task_type   = target_info["type"]
            logger.info("  Detected task type: %s", task_type)

            # ------------------------------------------------------------------
            # 3. Store target information
            # ------------------------------------------------------------------
            DATASTORE["target_info"]      = target_info
            DATASTORE["task_type"]        = task_type
            DATASTORE["target_stats"]     = target_stats
            DATASTORE["feature_strategy"] = feature_strategy

            if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
                y_binary = target_info["binary_target"]
                DATASTORE["y_binary"] = y_binary
                DATASTORE["y"]        = y_binary

                if "regression_target" in target_info:
                    DATASTORE["y_regression"] = target_info["regression_target"]
                if "regression_target_transformed" in target_info:
                    DATASTORE["y_regression_transformed"] = target_info["regression_target_transformed"]
                if "classification_target" in target_info:
                    DATASTORE["y_classification"] = target_info["classification_target"]

                y_align = y_binary

            elif "processed_target" in target_info:
                y_align = target_info["processed_target"]
                DATASTORE["y"]         = y_align
                DATASTORE["y_aligned"] = y_align
            else:
                y_align = target_series.dropna()
                DATASTORE["y"]         = y_align
                DATASTORE["y_aligned"] = y_align

            # ------------------------------------------------------------------
            # 4. Prepare features (X)
            # ------------------------------------------------------------------
            X_original = df.drop(columns=[target_col])
            X_original = X_original.loc[y_align.index]
            DATASTORE["X_original"] = X_original
            logger.info("  Feature matrix shape: %s", X_original.shape)

            # ------------------------------------------------------------------
            # 5. Warnings / messages
            # ------------------------------------------------------------------
            messages     = []
            warning_type = "info"

            if target_stats["missing_count"] > 0:
                msg = (f"Dropped {target_stats['missing_count']} rows "
                       f"({target_stats['missing_percentage']:.1f}%) with missing target values")
                messages.append(msg)
                logger.warning(msg)

            if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
                zero_ratio       = target_info.get("zero_ratio", 0)
                non_zero_samples = target_info.get("non_zero_samples", 0)
                msg = (f"Zero-inflated target detected: {zero_ratio:.1%} zeros. "
                       f"Using two-stage model approach with {non_zero_samples:,} non-zero samples.")
                messages.append(msg)
                warning_type = "warning"
                logger.warning(msg)

            elif task_type in ["binary_classification", "multiclass_classification"]:
                if "processed_target" in target_info:
                    class_counts   = target_info["processed_target"].value_counts()
                    if len(class_counts) > 0:
                        majority_ratio = class_counts.iloc[0] / len(target_info["processed_target"])
                        if majority_ratio > 0.8:
                            msg = f"Class imbalance detected: majority class is {majority_ratio:.1%} of data"
                            messages.append(msg)
                            logger.warning(msg)

            if len(y_align) < 100:
                msg = f"Small dataset after cleaning: {len(y_align)} samples"
                messages.append(msg)
                warning_type = "warning"
                logger.warning(msg)

            DATASTORE["target_messages"] = {"messages": messages, "type": warning_type}
            for msg in messages:
                flash(msg, warning_type)

            # ------------------------------------------------------------------
            # 6. Target summary
            # ------------------------------------------------------------------
            target_summary = {
                "column":       target_col,
                "task_type":    task_type,
                "original_rows": target_stats["total_rows"],
                "final_rows":   len(y_align),
                "dropped_rows": target_stats["total_rows"] - len(y_align),
                "zero_percentage": target_info.get("zero_ratio", 0) * 100,
                "unique_values": target_stats["unique_values"],
                "dtype":        target_stats["dtype"]
            }

            if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
                target_summary["non_zero_samples"] = target_info.get("non_zero_samples", 0)
                target_summary["transformation"]   = target_info.get("transformation", "none")

            DATASTORE["target_summary"] = target_summary

            return redirect(url_for("fe_auto"))

        except Exception as e:
            error_msg = str(e)
            logger.error("ERROR in target selection: %s", error_msg, exc_info=True)

            if "memory" in error_msg.lower():
                error_msg = "Memory error. Try a smaller dataset or sample your data first."
            elif "conversion" in error_msg.lower() or "numeric" in error_msg.lower():
                error_msg = (f"Cannot convert target column '{target_col}' to appropriate format. "
                             f"Try selecting a different column.")
            elif "not enough" in error_msg.lower() or "insufficient" in error_msg.lower():
                error_msg = ("Not enough valid data points after cleaning. "
                             "Try a different target column or enable auto-cleaning.")

            return render_template(
                "target.html",
                columns=df.columns.tolist(),
                column_analysis=column_analysis,
                dataset_size=dataset_size,
                is_large_dataset=is_large_dataset,
                error=f"Error processing target column: {error_msg}"
            )

    # GET
    previous_target = DATASTORE.get("target_column", "")

    if not column_analysis:
        column_analysis             = get_column_analysis(df)
        DATASTORE["column_analysis"] = column_analysis

    for col_info in column_analysis:
        col_name = col_info["name"]
        if col_name in df.columns:
            series = df[col_name]
            if is_numeric_dtype(series):
                zero_ratio = (series == 0).mean()
                if zero_ratio > 0.3:
                    col_info["is_zero_inflated"] = True
                    col_info["zero_percentage"]  = f"{zero_ratio*100:.1f}%"
                    if 0.3 < zero_ratio < 0.9:
                        non_zero = series[series != 0]
                        if len(non_zero) > 0 and non_zero.nunique() > 5:
                            col_info["suggested"] = True
                            col_info["reason"]    = (
                                f"Good for zero-inflated modeling "
                                f"({zero_ratio*100:.1f}% zeros, {non_zero.nunique()} unique non-zero values)"
                            )
                else:
                    col_info["is_zero_inflated"] = False

    return render_template(
        "target.html",
        columns=df.columns.tolist(),
        column_analysis=column_analysis,
        dataset_size=dataset_size,
        is_large_dataset=is_large_dataset,
        previous_target=previous_target,
        auto_clean_default=DATASTORE.get("auto_clean_applied", False)
    )


@app.route("/fe_auto", methods=["GET"])
def fe_auto():
    """Feature engineering page with zero-inflated support."""
    X_original    = DATASTORE.get("X_original")
    task_type     = DATASTORE.get("task_type")
    target_column = DATASTORE.get("target_column")
    target_info   = DATASTORE.get("target_info")

    if X_original is None:
        flash("Missing feature data. Please select a target column first.", "error")
        return redirect(url_for("select_target"))

    y_display       = None
    display_message = ""

    if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
        y_binary = DATASTORE.get("y_binary")
        if y_binary is None and target_info is not None:
            y_binary = target_info.get("binary_target")
        if y_binary is not None:
            y_display       = y_binary
            display_message = "Zero-inflated target detected. Using binary target for feature engineering."
    else:
        y_display = DATASTORE.get("y_aligned")
        if y_display is None:
            y_display = DATASTORE.get("y")
        if y_display is not None:
            display_message = "Regular target detected."

    if y_display is None:
        flash("Missing target data. Please go back and select a target column.", "error")
        return redirect(url_for("select_target"))

    zero_inflated_info = None
    if task_type in ["zero_inflated_regression", "zero_inflated_classification"] and target_info:
        zero_inflated_info = {
            "zero_ratio":       target_info.get("zero_ratio", 0),
            "non_zero_samples": target_info.get("non_zero_samples", 0),
            "transformation":   target_info.get("transformation", "none")
        }

    PROCESS_STATUS.update({
        "progress": 0, "stage": "idle", "operation": "",
        "message": "", "done": False, "error": None
    })

    return render_template(
        "fe_auto.html",
        task_type=task_type,
        target_column=target_column,
        x_shape=X_original.shape,
        y_length=len(y_display),
        zero_inflated_info=zero_inflated_info,
        display_message=display_message,
        has_data=True
    )


@app.route('/fe_auto/start', methods=['POST'])
def fe_auto_start():
    """Start feature engineering with proper y value."""
    global PROCESS_STATUS

    if PROCESS_STATUS.get("stage") in ["running", "complete"]:
        return jsonify({"status": "ignored", "message": "Already running or completed"})

    X_original  = DATASTORE.get("X_original")
    task_type   = DATASTORE.get("task_type")
    target_info = DATASTORE.get("target_info")

    logger.info("FE_AUTO_START: task_type=%s, target_info_exists=%s", task_type, target_info is not None)

    if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
        y = DATASTORE.get("y_binary")
        if y is None and target_info is not None:
            y = target_info.get("binary_target")
    else:
        y = DATASTORE.get("y_aligned")
        if y is None:
            y = DATASTORE.get("y")

    if X_original is None or y is None:
        logger.error("Missing data — X_original=%s, y=%s, task_type=%s, DATASTORE keys=%s",
                     X_original is not None, y is not None, task_type, list(DATASTORE.keys()))
        return jsonify({
            "status":  "error",
            "message": "Missing data. Please go back and select a target column first."
        }), 400

    PROCESS_STATUS = {
        "progress": 0, "stage": "running", "operation": "Starting",
        "message": "Starting feature engineering", "done": False, "error": None
    }

    logger.info("Starting feature engineering — X shape: %s, y length: %d", X_original.shape, len(y))

    threading.Thread(
        target=run_feature_engineering_async,
        args=(X_original, y, task_type),
        daemon=True
    ).start()

    return jsonify({"status": "started", "message": "Feature engineering started"})


@app.route("/feature_selection", methods=["GET", "POST"])
def feature_selection():
    X_processed = DATASTORE.get("X_processed")
    y_aligned   = DATASTORE.get("y_aligned")
    task_type   = DATASTORE.get("task_type")

    if X_processed is None or y_aligned is None:
        return redirect(url_for("fe_auto"))

    if request.method == "POST":
        selection_method = request.form.get("selection_method", "auto")
        n_features_raw   = request.form.get("n_features", "20")
        generate_viz     = request.form.get("generate_viz", "no") == "yes"

        try:
            n_features = int(n_features_raw) if n_features_raw.isdigit() else 'auto'
        except Exception:
            n_features = 'auto'

        selected_features, feature_scores, method_used, importance_img = intelligent_feature_selection(
            X_processed, y_aligned, task_type, n_features
        )

        DATASTORE["feature_importance_img"] = importance_img if generate_viz else None
        DATASTORE.update({
            "selected_features": selected_features,
            "feature_scores":    feature_scores,
            "selection_method":  method_used,
            "generate_viz":      generate_viz
        })

        return redirect(url_for("model_training_ui"))

    feature_stats = []
    for col in X_processed.columns:
        if is_numeric_dtype(X_processed[col]):
            corr = np.corrcoef(X_processed[col].fillna(0), y_aligned)[0, 1]
            feature_stats.append({
                "name": col,
                "type": "numeric",
                "correlation_with_target": f"{corr:.3f}" if not np.isnan(corr) else "N/A"
            })
        else:
            feature_stats.append({"name": col, "type": "other", "correlation_with_target": "N/A"})

    total_features = len(X_processed.columns)

    return render_template(
        "feature_selection.html",
        total_features=total_features,
        default_n_features=total_features,
        feature_stats=feature_stats[:20],
        task_type=task_type,
        has_data=True
    )


@app.route("/model_training", methods=["GET"])
def model_training_ui():
    if DATASTORE.get("X_processed") is None:
        return redirect(url_for("feature_selection"))

    task_type         = DATASTORE.get("task_type", "unknown")
    selected_features = DATASTORE.get("selected_features", [])
    X_processed       = DATASTORE.get("X_processed")

    if X_processed is not None and not selected_features:
        selected_features             = list(X_processed.columns)
        DATASTORE["selected_features"] = selected_features

    n_features           = len(selected_features)
    feature_importance_img = DATASTORE.get("feature_importance_img")
    generate_viz         = DATASTORE.get("generate_viz", False)

    if task_type == "regression" and not feature_importance_img and generate_viz:
        try:
            y_aligned    = DATASTORE.get("y_aligned")
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns if X_processed is not None else []

            if X_processed is not None and y_aligned is not None and len(numeric_cols) > 0:
                correlations = {}
                for col in numeric_cols[:15]:
                    try:
                        corr = X_processed[col].corr(y_aligned)
                        if not np.isnan(corr):
                            correlations[col] = abs(corr)
                    except Exception:
                        pass

                if correlations:
                    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
                    fig, ax      = plt.subplots(figsize=(8, 4))
                    labels       = [f[:20] + "..." if len(f) > 20 else f for f, _ in top_features]
                    scores       = [s for _, s in top_features]
                    y_pos        = np.arange(len(labels))

                    ax.barh(y_pos, scores)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(labels)
                    ax.invert_yaxis()
                    ax.set_xlabel("Correlation (absolute)")
                    ax.set_title("Top Feature Correlations with Target")
                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", dpi=80)
                    plt.close(fig)
                    buf.seek(0)
                    feature_importance_img = base64.b64encode(buf.read()).decode("utf-8")
                    DATASTORE["feature_importance_img"] = feature_importance_img

        except Exception as e:
            logger.warning("Quick visualization failed: %s", e)
            feature_importance_img = None

    is_zero_inflated = task_type in ["zero_inflated_regression", "zero_inflated_classification"]

    return render_template(
        "model_training.html",
        task_type=task_type,
        n_features=n_features,
        feature_importance_img=feature_importance_img,
        model_registry=MODEL_REGISTRY,
        selected_features=selected_features,
        is_zero_inflated=is_zero_inflated,
        target_info=DATASTORE.get("target_info", {}),
        datetime=datetime,
        now=datetime.now,
        format_number_filter=format_number_filter
    )


@app.route("/train_model", methods=["POST"])
def model_training():
    """Start model training with proper handling for zero-inflated."""
    if TRAINING_STATUS.get("running"):
        return jsonify({
            "started": False,
            "error":   "Training already in progress. Please wait or reset."
        }), 409

    X         = DATASTORE.get("X_processed")
    task_type = DATASTORE.get("task_type")
    target_info = DATASTORE.get("target_info")

    logger.info("MODEL_TRAINING: task_type=%s, X_shape=%s", task_type, X.shape if X is not None else None)

    if X is None:
        logger.error("X_processed not found. DATASTORE keys: %s", list(DATASTORE.keys()))
        return jsonify({"error": "Feature data not found. Please run feature engineering first."}), 400

    form_data = request.form.to_dict(flat=False)
    form_dict = process_form_data(form_data)

    models = form_dict.get('models', [])
    logger.info("Training config — mode: %s, tuning: %s, models: %d",
                form_dict.get('model_mode', 'auto'),
                form_dict.get('tuning_mode', 'auto'),
                len(models))

    if form_dict.get('model_mode') == 'manual' and not models:
        return jsonify({"error": "Please select at least one model for manual mode"}), 400

    param_dict = {}
    for key, value in form_dict.items():
        if '__' in key:
            parts = key.split('__')
            if len(parts) >= 3:
                new_key = f"{parts[0]}__{parts[2]}" if parts[1] == 'model' else '__'.join(parts[:2])
                param_dict[new_key] = value

    form_dict.update(param_dict)

    y = None
    if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
        if target_info is None:
            y_binary = DATASTORE.get("y_binary")
            if y_binary is not None:
                target_info = {
                    "type":                         task_type,
                    "binary_target":                y_binary,
                    "zero_ratio":                   (y_binary == 0).mean() if hasattr(y_binary, 'mean') else 0.5,
                    "non_zero_samples":             int(y_binary.sum()) if hasattr(y_binary, 'sum') else 0,
                    "regression_target":            DATASTORE.get("y_regression"),
                    "regression_target_transformed": DATASTORE.get("y_regression_transformed")
                }
            else:
                return jsonify({"error": "Cannot train zero-inflated model: missing target data"}), 400
        y = target_info
    else:
        y = DATASTORE.get("y_aligned")
        if y is None:
            y = DATASTORE.get("y")

    if y is None:
        return jsonify({"error": "Target data not found"}), 400

    selected_features = DATASTORE.get("selected_features", list(X.columns))

    TRAINING_STATUS.update({
        "running":       True,
        "done":          False,
        "current_model": None,
        "model_index":   0,
        "total_models":  0,
        "message":       "Starting training...",
        "started_at":    time.time(),
        "error":         None
    })

    global CURRENT_FORM_DATA
    CURRENT_FORM_DATA = form_dict

    try:
        threading.Thread(
            target=run_training,
            args=(X, y, task_type, selected_features, form_dict),
            daemon=True
        ).start()

        logger.info("Training thread started — X: %s, task: %s, features: %d",
                    X.shape, task_type, len(selected_features))

        return jsonify({
            "started":         True,
            "message":         "Training started successfully",
            "selected_models": len(models),
            "is_zero_inflated": task_type in ["zero_inflated_regression", "zero_inflated_classification"]
        }), 202

    except Exception as e:
        logger.error("Error starting training thread: %s", e, exc_info=True)
        TRAINING_STATUS.update({
            "running": False, "done": True,
            "message": f"Failed to start training: {str(e)}", "error": str(e)
        })
        return jsonify({"error": str(e)}), 500


@app.route("/results")
def results():
    """Display training results."""
    if not TRAINING_STATUS.get("done"):
        flash("Training not complete yet. Please wait.", "warning")
        return redirect(url_for("model_training_ui"))

    results_data = DATASTORE.get("training_results")
    if not results_data:
        flash("No training results found. Please train models first.", "error")
        return redirect(url_for("model_training_ui"))

    task_type              = DATASTORE.get("task_type", "")
    feature_importance_img = DATASTORE.get("feature_importance_img")
    selected_features      = DATASTORE.get("selected_features", [])
    is_zero_inflated       = isinstance(results_data, dict) and results_data.get("type") == "zero_inflated"

    model_comparison = format_model_comparison_for_display(results_data)

    best_model_key     = None
    best_model_name    = None
    best_model_metrics = {}

    if results_data.get("best_model"):
        best_model_key     = results_data["best_model"].get("key")
        best_model_name    = results_data["best_model"].get("name", "Unknown")
        best_model_metrics = results_data["best_model"].get("metrics", {})

    zero_inflated_display = format_zero_inflated_results(results_data) if is_zero_inflated else None

    processed_data_available = (DATASTORE.get("X_processed") is not None and
                                 DATASTORE.get("y_aligned") is not None)

    return render_template(
        'results.html',
        results=results_data,
        model_comparison=model_comparison,
        best_model_key=best_model_key,
        best_model_name=best_model_name,
        best_model_metrics=best_model_metrics,
        feature_importance_img=feature_importance_img,
        processed_data_available=processed_data_available,
        report_available=True,
        model_available=DATASTORE.get("best_model") is not None,
        task_type=task_type,
        selected_features=selected_features,
        is_zero_inflated=is_zero_inflated,
        zero_inflated_display=zero_inflated_display
    )


@app.route("/report")
def report_ui():
    training_results = DATASTORE.get("training_results")
    if training_results is None:
        return redirect(url_for("results"))

    report_data = {
        "dataset_info": {
            "original_shape":  DATASTORE["original_df"].shape if "original_df" in DATASTORE else "N/A",
            "processed_shape": DATASTORE["X_processed"].shape if "X_processed" in DATASTORE else "N/A"
        },
        "target_info": {
            "column":    DATASTORE.get("target_column"),
            "task_type": DATASTORE.get("task_type")
        },
        "feature_engineering": {
            "selected_features": DATASTORE.get("selected_features", []),
            "selection_method":  DATASTORE.get("selection_method", "N/A")
        },
        "model_results": training_results
    }

    return render_template("report.html", report=report_data)


@app.route("/download_processed_data")
def download_processed_data():
    """Download processed data as CSV."""
    try:
        X_processed = DATASTORE.get("X_processed")
        y_aligned   = DATASTORE.get("y_aligned")

        if X_processed is None or y_aligned is None:
            flash("No processed data available for download", "warning")
            return redirect(url_for('results'))

        df_processed = X_processed.copy()
        target_col   = DATASTORE.get("target_column")
        if target_col:
            df_processed[target_col] = y_aligned

        metadata_cols = [col for col in df_processed.columns if col.startswith('_')]
        if metadata_cols:
            df_processed = df_processed.drop(columns=metadata_cols)

        comment_cols = [col for col in df_processed.columns if '#' in col or 'comment' in col.lower()]
        if comment_cols:
            df_processed = df_processed.drop(columns=comment_cols)

        df_processed.columns = [
            str(col).replace('#', '').replace('_processed', '').strip()
            for col in df_processed.columns
        ]

        output = io.StringIO()
        df_processed.to_csv(output, index=False)
        output.seek(0)

        filename = f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )

    except Exception as e:
        logger.error("Error in download_processed_data: %s", e)
        flash(f"Error downloading processed data: {str(e)[:100]}", "danger")
        return redirect(url_for('results'))


@app.route("/download_report")
def download_report():
    """Download training report as JSON."""
    try:
        results = DATASTORE.get("training_results")
        if not results:
            flash("No training results available", "warning")
            return redirect(url_for('results'))

        fe_summary  = DATASTORE.get("feature_engineering_summary", {})
        report      = generate_comprehensive_report(results, fe_summary)
        report_json = json.dumps(report, indent=2, default=str)

        filename = f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return Response(
            report_json,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )

    except Exception as e:
        logger.error("Error in download_report: %s", e)
        flash(f"Error generating report: {str(e)[:100]}", "danger")
        return redirect(url_for('results'))


@app.route("/download_model")
def download_model():
    """Download the best trained model as a pickle file."""
    best_pipeline = DATASTORE.get("best_model")

    if best_pipeline is None:
        flash("No trained model available for download")
        return redirect(url_for("results"))

    try:
        training_results = DATASTORE.get("training_results", {})

        if training_results.get("type") == "zero_inflated":
            stage2           = training_results.get("stage2_regression", {})
            best_model_name  = stage2.get("best_model", {}).get("name", "best_model") if stage2 else "best_model"
        else:
            best_model_name  = training_results.get("best_model", {}).get("name", "best_model")

        filename = best_model_name.lower().replace(" ", "_").replace("/", "_") + ".pkl"

        buf = io.BytesIO()
        joblib.dump(best_pipeline, buf)
        buf.seek(0)

        return send_file(buf, as_attachment=True, download_name=filename,
                         mimetype="application/octet-stream")

    except Exception as e:
        flash(f"Error downloading model: {str(e)}")
        return redirect(url_for("results"))


@app.route("/api/models/<task_group>")
def get_available_models(task_group):
    """Get available models for a specific task group."""
    normalized_group = None
    if "classification" in task_group.lower():
        normalized_group = "classification"
    elif "regression" in task_group.lower():
        normalized_group = "regression"

    if normalized_group not in MODEL_REGISTRY:
        return jsonify({
            "success": False,
            "error":   f"Invalid task group: {task_group}. Available: {list(MODEL_REGISTRY.keys())}"
        }), 400

    is_zero_inflated = "zero_inflated" in task_group.lower()
    models_info      = []

    if is_zero_inflated:
        for model_key, cfg in MODEL_REGISTRY.get("classification", {}).items():
            models_info.append({
                "key": model_key, "label": cfg["label"], "task": "classification",
                "description": "Classification model for zero detection",
                "param_count": len(cfg.get("params", {})),
                "stage": "stage1", "stage_name": "Zero Detection",
                "params": cfg.get("params", {})
            })
        for model_key, cfg in MODEL_REGISTRY.get("regression", {}).items():
            models_info.append({
                "key": model_key, "label": cfg["label"], "task": "regression",
                "description": "Regression model for value prediction",
                "param_count": len(cfg.get("params", {})),
                "stage": "stage2", "stage_name": "Value Prediction",
                "params": cfg.get("params", {})
            })
    else:
        for model_key, cfg in MODEL_REGISTRY[normalized_group].items():
            models_info.append({
                "key": model_key, "label": cfg["label"], "task": normalized_group,
                "description": f"Sklearn {cfg['label']} model",
                "param_count": len(cfg.get("params", {})),
                "stage": "single", "stage_name": "Single Stage",
                "params": cfg.get("params", {})
            })

    return jsonify({
        "success":          True,
        "models":           models_info,
        "task_group":       task_group,
        "is_zero_inflated": is_zero_inflated
    })


@app.route("/api/model_params/<task_type>/<model_key>")
def api_model_parameters(task_type, model_key):
    """Get all parameters for a specific model, formatted for UI controls."""
    try:
        model_info    = None
        found_in_task = None

        for task_group in MODEL_REGISTRY:
            if model_key in MODEL_REGISTRY[task_group]:
                model_info    = MODEL_REGISTRY[task_group][model_key]
                found_in_task = task_group
                break

        if not model_info:
            return jsonify(success=False, error=f"Model {model_key} not found"), 404

        raw_params        = model_info.get("params", {})
        normalized_params = normalize_params(model_key, raw_params)
        formatted_params  = []

        for param_info in normalized_params:
            param_name  = param_info["name"]
            simple_name = param_name.replace("model__", "")

            param_info.update({
                "simple_name":  simple_name,
                "display_name": " ".join(w.capitalize() for w in simple_name.split("_")),
                "description":  f"Parameter: {simple_name}"
            })

            if simple_name in ["copy_X", "fit_intercept", "positive", "bootstrap", "shrinking", "probability"]:
                param_info["type"] = "bool"
                param_info.setdefault("options", [True, False])
                param_info.setdefault("default", True if simple_name in ["fit_intercept", "copy_X"] else False)

            for num_type in ["int", "float"]:
                if param_info["type"] == num_type:
                    cast = int if num_type == "int" else float
                    for k in ["min", "max", "step", "default"]:
                        if k in param_info and param_info[k] is not None:
                            try:
                                param_info[k] = cast(param_info[k])
                            except Exception:
                                pass

            formatted_params.append(param_info)

        return jsonify({
            "success":      True,
            "model_key":    model_key,
            "model_label":  model_info.get("label", model_key),
            "task_type":    found_in_task,
            "parameters":   formatted_params,
            "total_params": len(formatted_params)
        })

    except Exception as e:
        logger.error("ERROR in api_model_parameters: %s", e, exc_info=True)
        return jsonify(success=False, error=str(e)), 500


@app.route("/api/stop_training", methods=["POST"])
def stop_training():
    """Stop ongoing training."""
    global TRAINING_STATUS
    TRAINING_STATUS.update({
        "running": False, "done": True,
        "message": "Training stopped by user", "error": "Stopped by user"
    })
    return jsonify({"success": True, "message": "Training stopped"})


@app.route("/api/cancel_training", methods=["POST"])
def cancel_training():
    """Cancel ongoing training."""
    global TRAINING_STATUS
    TRAINING_STATUS.update({
        "running": False, "done": True,
        "message": "Training cancelled by user", "error": "Cancelled by user"
    })
    return jsonify({"success": True, "message": "Training cancelled"})


@app.route("/api/training_status")
def training_status():
    """Get current training status with detailed information."""
    try:
        status = TRAINING_STATUS.copy()

        if DATASTORE.get("training_results"):
            results   = DATASTORE["training_results"]
            task_type = DATASTORE.get("task_type", "")

            if task_type in ["zero_inflated_regression", "zero_inflated_classification"]:
                if isinstance(results, dict) and results.get("type") == "zero_inflated":
                    stage1_done = results.get("stage1_classification") is not None
                    stage2_done = results.get("stage2_regression") is not None

                    if not stage1_done:
                        status["current_stage"] = "classification"
                        status["stage_progress"] = {
                            "classification": {
                                "percentage":            status.get("progress", 0) * 100,
                                "status":                "running" if status.get("running") else "pending",
                                "message":               status.get("message", "Starting zero detection"),
                                "current_model":         status.get("current_model"),
                                "current_model_info":    f"Training {get_model_display_name(status.get('current_model', ''))}",
                                "current_model_progress": status.get("progress", 0),
                                "completed":             status.get("model_index", 0),
                                "total":                 status.get("total_models", 0)
                            }
                        }
                    elif stage1_done and not stage2_done:
                        status["current_stage"] = "regression"
                        status["stage_progress"] = {
                            "regression": {
                                "percentage":            status.get("progress", 0) * 100,
                                "status":                "running" if status.get("running") else "pending",
                                "message":               status.get("message", "Starting value prediction"),
                                "current_model":         status.get("current_model"),
                                "current_model_info":    f"Training {get_model_display_name(status.get('current_model', ''))}",
                                "current_model_progress": status.get("progress", 0),
                                "completed":             status.get("model_index", 0),
                                "total":                 status.get("total_models", 0)
                            }
                        }
                    else:
                        status["current_stage"] = "completed"
                        status["stage_progress"] = {
                            "classification": {"percentage": 100, "status": "completed"},
                            "regression":     {"percentage": 100, "status": "completed"}
                        }

        if status.get("current_model"):
            model_key = status["current_model"]
            status["current_model_info"] = f"Training {get_model_display_name(model_key)}"
            if status.get("model_index") and status.get("total_models"):
                status["current_model_progress"] = status["model_index"] / status["total_models"]

        for key, value in status.items():
            if isinstance(value, (np.integer, np.floating)):
                status[key] = float(value)
            elif isinstance(value, np.ndarray):
                status[key] = value.tolist()
            elif isinstance(value, pd.Timestamp):
                status[key] = value.isoformat()

        return jsonify(status)

    except Exception as e:
        logger.error("Error getting training status: %s", e)
        return jsonify({"running": False, "done": False, "error": str(e), "message": "Error getting status"})


@app.route("/api/processing_status")
def processing_status():
    import state
    return jsonify(state.PROCESS_STATUS)


@app.route("/api/fe_progress")
def api_fe_progress():
    """API endpoint for feature engineering progress."""
    feature_metadata = DATASTORE.get("feature_metadata", {})
    processing_steps = feature_metadata.get("processing_steps", [])
    total_steps      = 8
    completed_steps  = len([s for s in processing_steps if s.get("status") == "completed"])
    progress         = min(100, int((completed_steps / total_steps) * 100)) if total_steps > 0 else 0

    return jsonify({
        "progress":        progress,
        "completed_steps": completed_steps,
        "total_steps":     total_steps,
        "steps":           processing_steps,
        "is_complete":     DATASTORE.get("X_processed") is not None
    })


@app.route("/api/column_stats/<column_name>")
def column_stats(column_name):
    """API endpoint for column statistics."""
    df = DATASTORE.get("current_df")
    if df is None or column_name not in df.columns:
        return jsonify({"error": "Column not found"})

    col_data = df[column_name]
    col_type = detect_column_type(col_data)

    stats = {
        "name":               column_name,
        "type":               col_type,
        "missing_count":      int(col_data.isna().sum()),
        "missing_percentage": float(col_data.isna().mean() * 100),
        "unique_count":       int(col_data.nunique()),
        "sample_values":      col_data.dropna().head(10).tolist()
    }

    if col_type in ["numeric", "numeric_skewed", "numeric_discrete"]:
        numeric_data = safe_numeric_conversion(col_data.dropna())
        if len(numeric_data) > 0:
            stats.update({
                "min":      float(numeric_data.min()),
                "max":      float(numeric_data.max()),
                "mean":     float(numeric_data.mean()),
                "median":   float(numeric_data.median()),
                "std":      float(numeric_data.std()),
                "skewness": float(numeric_data.skew())
            })

    return jsonify(stats)


@app.route("/api/reset_progress", methods=["POST"])
def reset_progress():
    global PROCESS_STATUS
    PROCESS_STATUS = {
        "progress": 0, "stage": "starting", "operation": "",
        "message": "", "done": False, "error": None
    }
    return jsonify({"status": "reset"})


@app.route('/api/fe_summary')
def fe_summary():
    summary = DATASTORE.get("fe_summary")
    if not summary:
        return jsonify({"ready": False})
    return jsonify({"ready": True, **summary})


@app.route("/quick_clean", methods=["POST"])
def quick_clean():
    """Quick cleaning on selected columns — safe, accurate, UI-friendly."""
    try:
        df         = DATASTORE.get("current_df")
        target_col = request.form.get("target_column")

        if df is None or not target_col:
            return jsonify({"success": False, "error": "No dataset or target column selected"})

        df           = df.copy()
        rows_before  = len(df)

        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            df.drop(columns=empty_cols, inplace=True)

        df          = df[df[target_col].notna()]
        rows_dropped = rows_before - len(df)

        feature_cols = [c for c in df.columns if c != target_col]

        numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df[col].isna().any():
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "missing")

        DATASTORE["cleaned_df"] = df
        DATASTORE["current_df"] = df

        msg = (f"Dropped {rows_dropped} row(s) with missing target values"
               if rows_dropped > 0 else "No rows dropped. Dataset already clean.")

        return jsonify({
            "success":   True,
            "message":   msg,
            "new_shape": f"{df.shape[0]} rows × {df.shape[1]} columns"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/skip_processing")
def skip_processing():
    DATASTORE["processing_complete"] = True
    return redirect(url_for("dataset_overview"))


@app.route("/upload_status", methods=["GET"])
def upload_status():
    status = "complete" if DATASTORE.get("current_df") is not None else "processing"
    return jsonify({"status": status})


@app.route("/progress")
def progress_monitoring():
    status = "complete" if DATASTORE.get("processing_complete") else "processing"
    return jsonify({"status": status})


@app.route("/model_comparison")
def model_comparison():
    results = DATASTORE.get("training_results")
    if results is None:
        return redirect(url_for("model_training_ui"))

    return render_template(
        "model_comparison.html",
        model_metrics=results.get("test_metrics", {}),
        cv_scores=results.get("cv_scores", {}),
        best_model=results.get("best_model", {}).get("name", ""),
        feature_importance=results.get("feature_importance", {}),
        metrics_list=(list(results.get("test_metrics", {}).values())[0].keys()
                      if results.get("test_metrics") else [])
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/favicon.ico")
def favicon():
    return '', 204


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    try:
        return render_template("error.html", error="Page not found"), 404
    except Exception:
        return jsonify({"error": "404 - Page not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    try:
        return render_template("error.html", error="Internal server error"), 500
    except Exception:
        return jsonify({"error": "500 - Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point (dev only — production uses wsgi.py + gunicorn/waitress)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
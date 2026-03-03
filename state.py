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

PROCESS_STATUS = {
    "progress": 0,
    "stage": "idle",   # idle | dropping_index | imputing | encoding | scaling | selection | complete | error
    "operation": "",
    "message": "",
    "done": False,
    "error": None
}
TRAINING_STATUS = {
    "running": False,
    "done": False,
    "current_model": None,
    "model_index": 0,
    "total_models": 0,
    "models_completed": 0,
    "models_remaining": 0,
    "message": "Idle",
    "started_at": None,
    "stage": None,
    "stage_number": None,
    "progress": 0
}
LINEAR_MODELS = {
    "linear_regression",
    "ridge",
    "lasso",
    "logistic_regression"
}
NON_LINEAR_MODELS = {
    "random_forest_regressor",
    "random_forest_classifier",
    "gradient_boosting_regressor",
    "gradient_boosting_classifier",
    "svr",
    "svm",
    "knn_regressor",
    "knn_classifier",
    "adaboost_regressor",
    "adaboost_classifier"
}
MODEL_COST = {
    "linear_regression": "cheap",
    "ridge": "cheap",
    "lasso": "cheap",
    "logistic_regression": "cheap",

    "knn_classifier": "medium",
    "knn_regressor": "medium",

    "svm": "expensive",
    "svr": "expensive",

    "random_forest_regressor": "expensive",
    "random_forest_classifier": "expensive",

    "gradient_boosting_regressor": "medium",
    "xgboost": "medium"
}
TRAINING_RESULTS = None
CURRENT_FORM_DATA = None
DATASTORE = {}
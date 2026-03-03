from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans


MODEL_REGISTRY = {

    # =====================================================
    # REGRESSION
    # =====================================================
    "regression": {

        "linear_regression": {
            "label": "Linear Regression",
            "model": LinearRegression,
            "params": {
                "model__fit_intercept": [True, False],
                "model__copy_X": [True, False],
                "model__positive": [True, False]
            }
        },

        "ridge": {
            "label": "Ridge Regression",
            "model": Ridge,
            "params": {
                "model__alpha": {
                    "type": "float",
                    "min": 0.01,
                    "max": 50,
                    "step": 0.5,
                    "default": 1.0
                },
                "model__fit_intercept": [True, False],
                "model__copy_X": [True, False],
                "model__solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                "model__max_iter": {
                    "type": "int",
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                    "default": 1000
                },
                "model__tol": {
                    "type": "float",
                    "min": 1e-5,
                    "max": 1e-1,
                    "step": 0.0001,
                    "default": 1e-4
                }
            }
        },

        "lasso": {
            "label": "Lasso Regression",
            "model": Lasso,
            "params": {
                "model__alpha": {
                    "type": "float",
                    "min": 0.001,
                    "max": 1,
                    "step": 0.01,
                    "default": 0.1
                },
                "model__fit_intercept": [True, False],
                "model__copy_X": [True, False],
                "model__max_iter": {
                    "type": "int",
                    "min": 1000,
                    "max": 10000,
                    "step": 200,
                    "default": 5000
                },
                "model__tol": {
                    "type": "float",
                    "min": 1e-4,
                    "max": 1e-2,
                    "step": 0.0001,
                    "default": 1e-3
                },
                "model__selection": ['cyclic', 'random']
            }
        },

        "random_forest_regressor": {
            "label": "Random Forest Regressor",
            "model": RandomForestRegressor,
            "params": {
                "model__n_estimators": {
                    "type": "int",
                    "min": 50,
                    "max": 500,
                    "step": 50,
                    "default": 100
                },
                "model__max_depth": {
                    "type": "int",
                    "min": 3,
                    "max": 30,
                    "step": 3,
                    "default": None
                },
                "model__min_samples_split": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "step": 2,
                    "default": 2
                },
                "model__min_samples_leaf": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "default": 1
                },
                "model__max_features": ['sqrt', 'log2', None],
                "model__bootstrap": [True, False],
                "model__max_samples": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.1,
                    "default": None
                },
                "model__min_impurity_decrease": {
                    "type": "float",
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.01,
                    "default": 0.0
                }
            }
        },

        "gradient_boosting_regressor": {
            "label": "Gradient Boosting Regressor",
            "model": GradientBoostingRegressor,
            "params": {
                "model__n_estimators": {
                    "type": "int",
                    "min": 50,
                    "max": 300,
                    "step": 50,
                    "default": 100
                },
                "model__learning_rate": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.3,
                    "step": 0.05,
                    "default": 0.1
                },
                "model__max_depth": {
                    "type": "int",
                    "min": 3,
                    "max": 10,
                    "step": 1,
                    "default": 3
                },
                "model__min_samples_split": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "step": 2,
                    "default": 2
                },
                "model__min_samples_leaf": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "default": 1
                },
                "model__subsample": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.1,
                    "default": 1.0
                },
                "model__loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
                "model__criterion": ['friedman_mse', 'squared_error'],
                "model__max_features": ['sqrt', 'log2', None]
            }
        },

        "svr": {
            "label": "Support Vector Regressor",
            "model": SVR,
            "params": {
                "model__C": {
                    "type": "float",
                    "min": 0.1,
                    "max": 100,
                    "step": 0.5,
                    "default": 1.0
                },
                "model__kernel": ['rbf', 'linear', 'poly', 'sigmoid'],
                "model__gamma": ['scale', 'auto'],
                "model__degree": {
                    "type": "int",
                    "min": 2,
                    "max": 5,
                    "step": 1,
                    "default": 3
                },
                "model__epsilon": {
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.1
                },
                "model__shrinking": [True, False],
                "model__tol": {
                    "type": "float",
                    "min": 1e-5,
                    "max": 1e-1,
                    "step": 0.0001,
                    "default": 1e-3
                },
                "model__cache_size": {
                    "type": "int",
                    "min": 100,
                    "max": 1000,
                    "step": 100,
                    "default": 200
                }
            }
        },

        "knn_regressor": {
            "label": "KNN Regressor",
            "model": KNeighborsRegressor,
            "params": {
                "model__n_neighbors": {
                    "type": "int",
                    "min": 3,
                    "max": 25,
                    "step": 2,
                    "default": 5
                },
                "model__weights": ['uniform', 'distance'],
                "model__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                "model__leaf_size": {
                    "type": "int",
                    "min": 10,
                    "max": 50,
                    "step": 10,
                    "default": 30
                },
                "model__p": [1, 2],
                "model__metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            }
        },

        "adaboost_regressor": {
            "label": "AdaBoost Regressor",
            "model": AdaBoostRegressor,
            "params": {
                "model__n_estimators": {
                    "type": "int",
                    "min": 50,
                    "max": 300,
                    "step": 50,
                    "default": 100
                },
                "model__learning_rate": {
                    "type": "float",
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 1.0
                },
                "model__loss": ['linear', 'square', 'exponential']
            }
        }
    },

    # =====================================================
    # CLASSIFICATION
    # =====================================================
    "classification": {

        # Replace the logistic regression section in your models_registry.py:

"logistic_regression": {
    "label": "Logistic Regression",
    "model": LogisticRegression,
    "params": {
        "model__C": {
            "type": "float",
            "min": 0.01,
            "max": 10,
            "step": 0.1,
            "default": 1.0
        },
        "model__solver": {  # FIXED: Proper categorical parameter
            "type": "categorical",
            "values": ["lbfgs", "liblinear", "saga"],
            "default": "lbfgs"
        },
        "model__penalty": {  # FIXED: Proper categorical parameter
            "type": "categorical",
            "values": ["l2", "l1", "elasticnet", None],
            "default": "l2"
        },
        "model__max_iter": {
            "type": "int",
            "min": 100,
            "max": 5000,
            "step": 100,
            "default": 1000
        },
        "model__tol": {
            "type": "float",
            "min": 1e-5,
            "max": 1e-2,
            "step": 0.0001,
            "default": 1e-4
        },
        "model__class_weight": {  # FIXED: Proper categorical parameter
            "type": "categorical",
            "values": [None, "balanced"],
            "default": None
        },
        "model__l1_ratio": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "default": 0.5
        }
    }
},

        "random_forest_classifier": {
            "label": "Random Forest Classifier",
            "model": RandomForestClassifier,
            "params": {
                "model__n_estimators": {
                    "type": "int",
                    "min": 50,
                    "max": 500,
                    "step": 50,
                    "default": 100
                },
                "model__max_depth": {
                    "type": "int",
                    "min": 3,
                    "max": 30,
                    "step": 3,
                    "default": None
                },
                "model__min_samples_split": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "step": 2,
                    "default": 2
                },
                "model__min_samples_leaf": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "default": 1
                },
                "model__max_features": ['sqrt', 'log2', None],
                "model__bootstrap": [True, False],
                "model__max_samples": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.1,
                    "default": None
                },
                "model__min_impurity_decrease": {
                    "type": "float",
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.01,
                    "default": 0.0
                },
                "model__class_weight": [None, 'balanced', 'balanced_subsample'],
                "model__criterion": ['gini', 'entropy', 'log_loss']
            }
        },

        "gradient_boosting_classifier": {
            "label": "Gradient Boosting Classifier",
            "model": GradientBoostingClassifier,
            "params": {
                "model__n_estimators": {
                    "type": "int",
                    "min": 50,
                    "max": 300,
                    "step": 50,
                    "default": 100
                },
                "model__learning_rate": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.3,
                    "step": 0.05,
                    "default": 0.1
                },
                "model__max_depth": {
                    "type": "int",
                    "min": 3,
                    "max": 10,
                    "step": 1,
                    "default": 3
                },
                "model__min_samples_split": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "step": 2,
                    "default": 2
                },
                "model__min_samples_leaf": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "default": 1
                },
                "model__subsample": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.1,
                    "default": 1.0
                },
                "model__loss": ['log_loss', 'exponential'],
                "model__criterion": ['friedman_mse', 'squared_error'],
                "model__max_features": ['sqrt', 'log2', None]
            }
        },

        "svm": {
            "label": "Support Vector Machine",
            "model": SVC,
            "params": {
                "model__C": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10,
                    "step": 0.5,
                    "default": 1.0
                },
                "model__kernel": ['rbf', 'linear', 'poly', 'sigmoid'],
                "model__gamma": ['scale', 'auto'],
                "model__degree": {
                    "type": "int",
                    "min": 2,
                    "max": 5,
                    "step": 1,
                    "default": 3
                },
                "model__shrinking": [True, False],
                "model__probability": [True, False],
                "model__tol": {
                    "type": "float",
                    "min": 1e-5,
                    "max": 1e-1,
                    "step": 0.0001,
                    "default": 1e-3
                },
                "model__class_weight": [None, 'balanced'],
                "model__decision_function_shape": ['ovo', 'ovr'],
                "model__break_ties": [True, False]
            }
        },

        "knn_classifier": {
            "label": "KNN Classifier",
            "model": KNeighborsClassifier,
            "params": {
                "model__n_neighbors": {
                    "type": "int",
                    "min": 3,
                    "max": 25,
                    "step": 2,
                    "default": 5
                },
                "model__weights": ['uniform', 'distance'],
                "model__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                "model__leaf_size": {
                    "type": "int",
                    "min": 10,
                    "max": 50,
                    "step": 10,
                    "default": 30
                },
                "model__p": [1, 2],
                "model__metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            }
        },

        "adaboost_classifier": {
            "label": "AdaBoost Classifier",
            "model": AdaBoostClassifier,
            "params": {
                "model__n_estimators": {
                    "type": "int",
                    "min": 50,
                    "max": 300,
                    "step": 50,
                    "default": 100
                },
                "model__learning_rate": {
                    "type": "float",
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 1.0
                },
                "model__algorithm": ['SAMME', 'SAMME.R']
            }
        }
    }
}
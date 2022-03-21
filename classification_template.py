# ---- Import Packages ---------------------------------------------------------
import pandas as pd

# ---- Import Data -------------------------------------------------------------
df = pd.read_csv("")


# ---- Inspect Data ------------------------------------------------------------

# Data Shape
print(f"Shape: {df.shape}")

# Data Types
print(df.info())

# Proportion Missing
perc_missing = df.isna().mean(axis=0)
print(f"Percent Missing\n{perc_missing}")

# Data Summary Numerical
print(df.describe())


# ---- Data Cleaning -----------------------------------------------------------

# Drop columns with nulls over a specified percentage
over_threshold = perc_missing[perc_missing >= 0.4]
df = df.drop(over_threshold.index, axis=1)

# ---- Pipeline Column Tranformation -------------------------------------------


def transform_and_split(df, y_col_name, test_size=0.1, val_size=0.2, SEED=830):
    """
    Transforms data by standardizing numerical features and onehot encoding 
    categorical features. Missing values are handled using mean imputation (numerical)
    and most frequent imputation (categorical). 

    Data is split into train, validation, and testing sets. The transformer is then
    fit on the training data and all sets are then transformed. 

    PARAMS: 
        df: dataframe of all data
        y_col_name (str): the name of the columns to use as the target feature
        test_size: portion of total data to split off as a test set
        val_size: proportion of non-test data to use as validation data
        SEED: used to set a random state for reproducability

    RETURNS: 
        X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p
    """
    # ---- Imports ----
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    # ---- Separate X and y features ----
    X = df.drop(columns=[y_col_name])
    y = df[y_col_name]

    # ---- Extract columns by type ----
    cat_cols = X.select_dtypes(exclude="number").columns
    num_cols = X.select_dtypes(include="number").columns

    # ---- Create pipelines ----
    # Numeric Features
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
    )
    # Categorical Features
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    # Combine Pipelines
    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )

    # ---- Split data ----
    # Split off Test Data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=SEED
    )
    # Split temp into Train and Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=val_size, random_state=830
    )

    # ---- Fit Transformer on X Train ----
    fit_processor = full_processor.fit(X_train)

    # ---- Transform all X data splits ----
    X_train_p = fit_processor.transform(X_train)
    X_val_p = fit_processor.transform(X_val)
    X_test_p = fit_processor.transform(X_test)

    # ---- Return Transformed and Split Data ----
    return X_train_p, X_val_p, X_test_p, y_train, y_val, y_test


# ---- Common Imports ----------------------------------------------------------
from sklearn.metrics import plot_confusion_matrix, accuracy, roc_auc_score
from sklearn.model_selection import GridSearchCV


# ---- Classification Algorithms -----------------------------------------------
X_train, X_val, X_test, y_train, y_val, y_test = transform_and_split()

# ---- Logistic Regression ----
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
y_pred = log_clf.predict(X_val)


# ---- K Nearest Neighbors ----
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
# Set up Grid Search
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)

# create GridSearch Object
grid = GridSearchCV(
    knn_clf, param_grid, cv=10, scoring="accuracy", return_train_score=False, verbose=1
)

# fitting the model for grid search
gs_knn = grid.fit(X_train, y_train)
print(gs_knn.best_params_)


# ---- Support Vector Machine ----
from sklearn.svm import SVC

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf", "poly", "sigmoid"],
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
gs_svc = grid.fit(X_train, y_train)

print(gs_svc.best_estimator_)

# ---- Random Forest Classifier ----
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "bootstrap": [True, False],
    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    "max_features": ["auto", "sqrt"],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "n_estimators": [200, 400, 600, 800],
}

rf_clf = RandomForestClassifier()
grid = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5)
gs_rf = grid.fit(X_train, y_train)

print(gs_rf.best_estimator_)

# ---- Keras Neural Network ----
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Load Data
target = to_categorical(y_train)
predictors = X_train
n_cols = predictors.shape[1]
auc_metric = keras.metrics.AUC()

# Specify the model
model = Sequential()
model.add(Dense(100, activation="relu", input_shape=(n_cols,)))
model.add(Dense(100, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=auc_metric)

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=3)

# Fit the model
model = model.fit(
    predictors,
    target,
    validation_split=0.3,
    epochs=30,
    callbacks=[early_stopping_monitor],
    verbose=1,
)

# ---- XGBoost ----
# conda install -c conda-forge xgboost
# import xgboost as xgb
# xgb_cl = xgb.XGBClassifier()


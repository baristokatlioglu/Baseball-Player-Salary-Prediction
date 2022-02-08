#Business Problem
#Salary information and career statistics for 1986
#for shared baseball players' salary estimates
#Can a machine learning project be realized?

# Variables
# AtBat: Number of hits with a baseball bat during the 1986-1987 season
# Hits: Number of hits in the 1986-1987 season
# HmRun: Most valuable hits in the 1986-1987 season
# Runs: The points he earned for his team in the 1986-1987 season
# RBI: Number of players a batter had jogged when he hit
# Walks: Number of mistakes made by the opposing player
# Years: Player's playing time in major league (years)
# CAtBat: Number of hits during a player's career
# CHits: The number of hits the player has taken throughout his career
# CHmRun: The player's most valuable hit during his career
# CRuns: Points earned by the player during his career
# CRBI: The number of players the player has made during his career
# CWalks: The number of mistakes the player has made to the opposing player during their career
# League: A factor with A and N levels showing the league in which the player played until the end of the season
# Division: A factor with levels E and W indicating the position played by the player at the end of 1986
# PutOuts: Helping your teammate in-game
# Assists: Number of assists made by the player in the 1986-1987 season
# Errors: Player's number of errors in the 1986-1987 season
# Salary: The salary of the player in the 1986-1987 season (over thousand)
# NewLeague: A factor with A and N levels showing the player's league at the start of the 1987 season

# !pip install xgboost
# !pip install lightgbm
# conda install -c conda-forge lightgbm
# !pip install catboost

import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV
from helpers.data_prep import *
from helpers.eda import *
warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
######
# Reading Data
######
def load_data():
    df = pd.read_csv("datasets/hitters.csv")
    return df
df = load_data()

#####
# EDA
#####
check_df(df)

# Salary  has 59 missing values, i will drop these values

#####
# Missing Values
#####
df.dropna(inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#####
# Outlier Analysis
#####
for col in num_cols:
    print(col, check_outlier(df, col))


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit(df[num_cols])
df_scores = clf.negative_outlier_factor_
df_scores[:5]
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

#The sharp decrease in slope decreases after the 5th value, I will accept this value as the threshold value.
th = np.sort(df_scores)[5]
df.drop(axis=0, labels = df[df_scores < th].index, inplace=True)


######
# Feature Engineering
######

df.columns

df["AVGAtBat"] = df["AtBat"] / df["CAtBat"]
df["AVGHits"] = df["Hits"] / df["CHits"]
df["AVGRun"] = df["HmRun"] / df["CHmRun"]
df["AVGRuns"] = df["Runs"] / df["CRuns"]
df["AVGRBI"] = df["RBI"] / df["CRBI"]
df["AVGWalks"] = df["Walks"] / df["CWalks"]
df["AVGAssists"] = df["Assists"] / df["Years"]
df["AVGErrors"] = df["Errors"] / df["Years"]
df["AtBat_Years"] = df["AtBat"] / df["Years"]
df["Hits_Years"] = df["Hits"] / df["Years"]
df["HmRun_Years"] = df["HmRun"] / df["Years"]
df["Runs_Years"] = df["Runs"] / df["Years"]
# Categorical variable expressing the player's experience
df["Years_CAT"] = pd.qcut(df["Years"], 4, labels=["Junior", "Senior", "Expert", "Pro"])
df["Years_CAT"]=df.Years_CAT.astype("O")
df["TotalAtBat/Years"] = (df["AtBat"] + df["CAtBat"]) / df["Years"]
df["TotalHits/Years"] = (df["Hits"] + df["CHits"]) / df["Years"]
df["TotalHmRun/Years"] = (df["HmRun"] + df["CHmRun"]) / df["Years"]
df["TotalRuns/Years"] = (df["Runs"] + df["CRuns"]) / df["Years"]
df["BattingAverage"] = df["Hits"] / df["AtBat"]
df["CBattingAverage"] = df["CHits"] / df["CAtBat"]
df["TotalChances"] = df["Assists"] + df["PutOuts"] + df["Errors"]


df.isnull().sum()
df["AVGRun"].fillna(0, inplace=True)
cat_cols,num_cols,cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df,col))

###
# Encoding
###
#LabelEncoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

# One Hot Encoder
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, True)

####
# Scaling
####
# Since I do Outlier analysis, I use Standard Scaler, not Robust Scaler.
# Standart Scaling
num_cols = [col for col in num_cols if "Salary" not in col]
std_scaler = StandardScaler()
df[num_cols] = std_scaler.fit_transform(df[num_cols])
df.head()

#####
# BASE MODELS
#####
# Since the number of data is small, I will not separate it as train and test.
y = df["Salary"]
X = df.drop("Salary", axis=1)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# RMSE: 257.0607 (LR)
# RMSE: 249.2322 (Ridge)
# RMSE: 253.0175 (Lasso)
# RMSE: 261.5007 (ElasticNet)
# RMSE: 263.6913 (KNN)
# RMSE: 399.6017 (CART)
# RMSE: 240.3881 (RF)
# RMSE: 433.6587 (SVR)
# RMSE: 235.8076 (GBM)
# RMSE: 267.4633 (XGBoost)
# RMSE: 252.1631 (LightGBM)
# RMSE: 234.7101 (CatBoost)


####
# RandomizedSearchCV
####

# RANDOM FORESTS
rf_model = RandomForestRegressor()
rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)
rf_random.fit(X, y)
rf_random.best_params_

# {'n_estimators': 488,
#  'min_samples_split': 5,
#  'max_features': 5,
#  'max_depth': 38}

######################################################
# Automated Hyperparameter Optimization
######################################################


cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [15,20,30,None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [3,5,8],
             "n_estimators": [500,800,1000]}

gbm_params = {"learning_rate": [0.01,0.1,0.001],
              "n_estimators" : [200,500,1000],
              "subsample" : [0.5, 0.7, 1],
              "min_samples_split" : [2, 5, 10],
              "max_depth" : [5,8,10,20]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}


regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ("GBM", GradientBoostingRegressor(), gbm_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

best_models = {}



for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model






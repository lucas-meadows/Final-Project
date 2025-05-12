import pandas as pd
import ast

odds_df = pd.read_csv("engineered_fight_features.csv")
fighter_df =pd.read_csv("kaggle_fighter_stats.csv")


# Converting fighter dataframe into a dictionary, where we can choose the correct dictionary features for our project
fighter_df["about_dict"] = fighter_df["about"].apply(ast.literal_eval)

about_df = fighter_df["about_dict"].apply(pd.Series)

# Selecting desired features
about_df = about_df[[
    "id", "name", "division", "gender", "Status",
    "Place of Birth", "Age", "Octagon Debut",
    "Height", "Reach", "Leg reach"
]]

final_df = about_df.copy()





# Pivoting our Odds data set so that rows with the same fight ID are joined
odds_df["fighter_role"] = odds_df.groupby("fight_id").cumcount()

odds_df = odds_df.pivot(index="fight_id", columns="fighter_role")

odds_df.columns = [f"{col[0]}_fighter{col[1]+1}" for col in odds_df.columns]

odds_df.reset_index(inplace=True)

odds_df.columns = [col.replace("fighter1", "fighter_A").replace("fighter2", "fighter_B") for col in odds_df.columns]

# Remove any duplicate rows (where it is same for fighter A and fighter B)
odds_df.drop(columns=[
    "time_to_fight_hrs_fighter_B",
    "avg_num_books_fighter_B",
    "large_movement_fighter_B"
], inplace=True)


# Merging the Odds dataset with thefighter dataset, removing any fights without complete data
odds_df["fighter_B_temp"] = odds_df["fighter_fighter_B"]

merged = odds_df.merge(final_df, how="left", left_on="fighter_fighter_A", right_on="name")
merged.drop(columns=["name"], inplace=True)
merged = merged.rename(columns={col: f"{col}_A" for col in final_df.columns if col != "name"})

merged = merged.merge(final_df, how="left", left_on="fighter_B_temp", right_on="name")
merged.drop(columns=["name", "fighter_B_temp"], inplace=True)
merged = merged.rename(columns={col: f"{col}_B" for col in final_df.columns if col != "name"})

# Dropping rows without both fighter's data
merged.dropna(subset=["division_A", "division_B"], inplace=True)





# Producing our classification model, we are using a Logistic Regression Model

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# renaming for convention, as any large movement in the market applies to A and B
merged = merged.rename(columns={"large_movement_fighter_A": "large_odds_movement"})

# Select features, removing those that risk data leakage
numeric_features = [
    'Age_A', 'Age_B', 'Height_A', 'Height_B', 'Reach_A', 'Reach_B',
    'start_prob_fighter_A', 'start_prob_fighter_B',
    'avg_spread_across_books_fighter_A', 'avg_spread_across_books_fighter_B',
    'avg_num_books_fighter_A', 'time_to_fight_hrs_fighter_A'
]

categorical_features = ['division_A', 'division_B', 'Status_A', 'Status_B']

X = merged[numeric_features + categorical_features]
y = merged["large_odds_movement"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop='first'), categorical_features)
])

clf = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Producing feature importance

# Group feature names
coef_summary = pd.DataFrame({
    "feature": all_features,
    "coef": coefs
})

# Group by original base feature (before underscores)
coef_summary["base_feature"] = coef_summary["feature"].str.extract(r"(^[^_]+_[^_]+)")  # e.g., 'division_A', 'Status_B'

# Sum absolute values by base feature to get a sense of importance
grouped = coef_summary.groupby("base_feature")["coef"].apply(lambda x: x.abs().sum()).sort_values(ascending=False)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(x=grouped.values, y=grouped.index)
plt.title("Feature Group Importance (Summed Abs Coefficients)")
plt.xlabel("Sum of Absolute Coefficients")
plt.tight_layout()
plt.show()


# Producing new model with k-fold cross validation

from sklearn.linear_model import LogisticRegressionCV

clf = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegressionCV(
        Cs=10,                      # Try 10 values for regularization strength
        cv=5,                       # 5-fold cross-validation internally
        penalty='l2',              # L2 regularization (default, stable)
        scoring='f1',              # Focus tuning on F1 score
        max_iter=1000,
        random_state=42
    ))
])

from sklearn.metrics import classification_report

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("train.csv")
df["age"] = df["age"].astype(int)
df = df.drop(columns=["ID", "age_desc"])

mapping = {"Viet Nam": "VietNam", "AmericanSamoa": "United States", "Hong Kong": "China"}
df["contry_of_res"] = df["contry_of_res"].replace(mapping)
df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
df["relation"] = df["relation"].replace({
    "?": "Others", "Relative": "Others",
    "Parent": "Others", "Health care professional": "Others"
})

def replace_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    median = df[col].median()
    df[col] = df[col].apply(
        lambda x: median if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR else x)
    return df

df = replace_outliers(df, "age")
df = replace_outliers(df, "result")

encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

models = {
    "Decision Tree": (DecisionTreeClassifier(random_state=42),
        {"criterion":["gini","entropy"],"max_depth":[None,10,20,30],
         "min_samples_split":[2,5,10],"min_samples_leaf":[1,2,4]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
        {"n_estimators":[50,100,200],"max_depth":[None,10,20],
         "min_samples_split":[2,5],"min_samples_leaf":[1,2]}),
    "XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss"),
        {"n_estimators":[50,100,200],"max_depth":[3,5,7],
         "learning_rate":[0.05,0.1,0.3]}),
}

best_model, best_score = None, 0
for name, (model, params) in models.items():
    rs = RandomizedSearchCV(model, params, n_iter=10, cv=5,
                            scoring="accuracy", random_state=42)
    rs.fit(X_train, y_train)
    print(f"{name}: {rs.best_score_:.3f}")
    if rs.best_score_ > best_score:
        best_model, best_score = rs.best_estimator_, rs.best_score_

y_pred = best_model.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Saved: best_model.pkl and encoders.pkl")
import joblib
import pandas as pd
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression

df = sns.load_dataset("titanic")

df =df[["pclass","sex","age","fare","survived"]]

x = df.drop("survived",axis =1)
y = df["survived"]

numeric_features = ["age","fare"]
categorical_features = ["pclass","sex"]

numeric_pipeline = Pipeline(
    steps = [
        ("imputer",SimpleImputer(strategy = "median")),
        ("scaler",StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps = [
        ("imputer",SimpleImputer(strategy = "most_frequent")),
        ("encoder",OrdinalEncoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ("num",numeric_pipeline,numeric_features),
        ("cat",categorical_pipeline,categorical_features)
    ]
)

pipeline = Pipeline(
    steps = [
        ("preprocessor",preprocessor),
        ("model",LogisticRegression())
    ]
)

pipeline.fit(x,y)

os.makedirs("model",exist_ok =True)
joblib.dump(pipeline,"model/pipeline.pkl")
print("Pipeline saved sucessfully")
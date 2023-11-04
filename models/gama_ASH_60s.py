from numpy import nan
from category_encoders.target_encoder import TargetEncoder
from category_encoders.ordinal import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from category_encoders.one_hot import OneHotEncoder

pipeline = Pipeline(
    [
        ("ord-enc", OrdinalEncoder(cols=[], drop_invariant=True)),
        ("oh-enc", OneHotEncoder(cols=[])),
        ("target_enc", TargetEncoder(cols=["id"])),
        ("imputation", SimpleImputer(strategy="median")),
        (
            "0",
            ExtraTreesRegressor(
                bootstrap=True,
                max_features=0.6500000000000001,
                min_samples_leaf=3,
                min_samples_split=20,
                n_estimators=100,
            ),
        ),
    ]
)

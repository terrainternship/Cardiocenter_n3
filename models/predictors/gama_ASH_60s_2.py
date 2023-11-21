from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from category_encoders.one_hot import OneHotEncoder
from numpy import nan
from category_encoders.target_encoder import TargetEncoder
from category_encoders.ordinal import OrdinalEncoder

pipeline = Pipeline(
    [
        ("ord-enc", OrdinalEncoder(cols=[], drop_invariant=True)),
        ("oh-enc", OneHotEncoder(cols=[])),
        ("target_enc", TargetEncoder(cols=[])),
        ("imputation", SimpleImputer(strategy="median")),
        ("1", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
        ("0", KNeighborsRegressor(n_neighbors=34, p=1, weights="distance")),
    ]
)

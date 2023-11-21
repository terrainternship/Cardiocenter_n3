from sklearn.cluster import FeatureAgglomeration
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.neighbors import KNeighborsRegressor
from category_encoders.one_hot import OneHotEncoder
from numpy import nan
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

pipeline = Pipeline(
    [
        ("ord-enc", OrdinalEncoder(cols=[], drop_invariant=True)),
        ("oh-enc", OneHotEncoder(cols=[])),
        ("target_enc", TargetEncoder(cols=[])),
        ("imputation", SimpleImputer(strategy="median")),
        ("1", FeatureAgglomeration(affinity="euclidean", linkage="average")),
        ("0", KNeighborsRegressor(n_neighbors=37, p=2, weights="distance")),
    ]
)

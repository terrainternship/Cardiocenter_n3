from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from numpy import nan
from sklearn.cluster import FeatureAgglomeration
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor


pipeline = Pipeline(
    [
        ('ord-enc', OrdinalEncoder(cols=[], drop_invariant=True)),
        ('oh-enc', OneHotEncoder(cols=[])),
        ('target_enc', TargetEncoder(cols=[])),
        ('imputation', SimpleImputer(strategy='median')),
        ('1', FeatureAgglomeration(affinity='manhattan', linkage='average')),
        ('0', KNeighborsRegressor(n_neighbors=93, p=1, weights='distance'))
    ]
)

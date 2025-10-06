from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def create_preprocessor(num_cols, cat_cols=None):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # imputação robusta
        ('scaler', StandardScaler())                    # padronização
    ])
    transformers = [('num', num_pipeline, num_cols)]

    if cat_cols:
        from sklearn.preprocessing import OneHotEncoder
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return preprocessor

def preview_transformation(preprocessor, df, n=5):
    """Mostra resultado do preprocessor para as n primeiras linhas."""
    transformed = preprocessor.fit_transform(df)
    transformed_df = pd.DataFrame(transformed)
    print("\nShape original:", df.shape)
    print("Shape transformado:", transformed_df.shape)
    print(transformed_df.head(n))
    return transformed_df

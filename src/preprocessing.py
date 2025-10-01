from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def create_preprocessor(num_cols):
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
    ])
    return preprocessor

def preview_transformation(preprocessor, df, n=5):
    """Mostra resultado do preprocessor para as n primeiras linhas."""
    transformed = preprocessor.fit_transform(df)
    # transformed_df = pd.DataFrame(transformed)
    # print("\nShape original:", df.shape)
    # print("Shape transformado:", transformed_df.shape)
    # print(transformed_df.head(n))
    return transformed
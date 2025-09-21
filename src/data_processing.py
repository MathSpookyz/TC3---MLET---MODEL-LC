import pandas as pd

file_path = r"C:\Users\Matheus\Downloads\archive\lung_cancer_dataset.csv"

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['patient_id'])
    # imputação
    df['age'].fillna(df['age'].median(), inplace=True)
    df['pack_years'].fillna(df['pack_years'].median(), inplace=True)
    # engenharia de features
    df['exposure_count'] = df[['asbestos_exposure','secondhand_smoke_exposure','copd_diagnosis','family_history']]\
                            .apply(lambda x: sum([1 if v in ['Yes','High'] else 0 for v in x]), axis=1)
    return df

df = load_and_process_data(file_path)

# Resumo estatístico
print(df.describe(include='all'))   # estatísticas
print(df.info())                    # tipos de dados
print(df.head())                    # primeiras linhas
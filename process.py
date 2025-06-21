import pandas as pd

class DataProcessor:
    def __init__(self):
        self.df = None
        self.synthetic_df = None

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        self.df = self.preprocess_data(df)

    def preprocess_data(self, df):
        df = df.copy()
        df.dropna(axis=1, how='all', inplace=True)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        return df

    def set_synthetic_path(self, filepath):
        self.synthetic_df = pd.read_csv(filepath)
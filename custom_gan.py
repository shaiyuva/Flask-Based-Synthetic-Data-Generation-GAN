import pandas as pd
import numpy as np
import os

def generate_custom_gan(df, output_folder):
    synthetic = df.copy()
    for col in df.select_dtypes(include='number'):
        synthetic[col] += np.random.normal(0, 1, size=len(df))
    out_path = os.path.join(output_folder, 'synthetic_custom.csv')
    synthetic.to_csv(out_path, index=False)
    return synthetic
from ctgan import CTGAN
import pandas as pd
import os
real_df = pd.read_csv("./synthetic_custom.csv")
synth_df=pd.read_csv("C:\\Users\\shai\\PycharmProjects\\PythonProject2\\outputs\\synthetic_ctgan.csv")
"""
def generate_ctgan(df, output_folder):
    print(df)
    model = CTGAN(epochs=10)
    model.fit(df)
    synthetic = model.sample(len(df))
    out_path = os.path.join(output_folder, 'synthetic_ctgan.csv')
    synthetic.to_csv(out_path, index=False)
    return synthetic

synthetic_data = generate_ctgan(df,"./outputs")"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd

def generate_statistics_report(real_df, synth_df):
    real_stats = real_df.describe().to_html(classes='table table-bordered', index=True)
    synth_stats = synth_df.describe().to_html(classes='table table-bordered', index=True)
    return f"<h2>Real Data Stats</h2>{real_stats}<h2>Synthetic Data Stats</h2>{synth_stats}"

# report=generate_statistics_report(real_df,synth_df)
# print("Generated Synthetic Data:")
# print(report)

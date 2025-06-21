from ctgan import CTGAN
import os
def generate_ctgan(df, output_folder):
    print(df)
    model = CTGAN(epochs=10)
    model.fit(df)
    synthetic = model.sample(len(df))
    out_path = os.path.join(output_folder, 'synthetic_ctgan.csv')
    synthetic.to_csv(out_path, index=False)
    return synthetic

#synthetic_data = generate_ctgan(df,"./outputs")
#print("Generated Synthetic Data:")
#print(synthetic_data.head())

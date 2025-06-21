from flask import Flask, render_template, request, redirect, send_file, render_template_string
import pandas as pd
import os
from eda import generate_eda_report, get_statistical_summary
from ctgan_generator import generate_ctgan
from custom_gan import generate_custom_gan
from reports import generate_statistics_report, generate_visualizations

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
df = {}
syn_df = {}
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/')
# def upload_form():
#     return render_template_string(open('./upload.html').read())

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        global df
        df = pd.read_csv(filepath)
        eda_figs = generate_eda_report(df)
        eda_html_list = [fig.to_html(full_html=False) for fig in eda_figs]
        stats = get_statistical_summary(df)
        return render_template('generate.html', eda_html_list=eda_html_list, stats=stats, filename=file.filename)
    return redirect('/')

@app.route('/generate/ctgan', methods=['POST'])
def ctgan():
    global df
    global syn_df
    syn_df = generate_ctgan(df, "./outputs")
    return render_template('download.html')
    #return redirect('/')
    #print(df)
    #html_table = df.to_html(classes='table table-bordered', index=False)
    #return f"<h3>Uploaded and Read CSV:</h3>{html_table}"

@app.route('/generate/custom', methods=['POST'])
def generate_customgan():
    global df
    global syn_df
    syn_df = generate_custom_gan(df, "./outputs")
    return render_template('download.html')


@app.route('/compare', methods=['POST'])
def report_choice():
    return render_template('report_choice.html')

@app.route('/report/statistics', methods=['POST'])
def statistics():
    global df
    global syn_df
    print(df)
    print(syn_df)
    return generate_statistics_report(df, syn_df)

@app.route('/report/visualization', methods=['POST'])
def visualization():
    global df
    global syn_df
    print(df)
    print(syn_df)
    imageList = generate_visualizations(df, syn_df)
    return render_template('visualization.html', plots=imageList)



@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True, use_reloader=False, port=2001)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_statistics_report(real_df, synth_df):
    # Handle missing values
    real_df = real_df.fillna(real_df.mean(numeric_only=True))
    synth_df = synth_df.fillna(synth_df.mean(numeric_only=True))

    numeric_cols = real_df.select_dtypes(include='number').columns
    report_html = ""

    def stats_for_column(df, col):
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        mode = series.mode()
        mode_val = mode.iloc[0] if not mode.empty else "N/A"
        return {
            "Mean": round(series.mean(), 2),
            "Median": round(series.median(), 2),
            "Mode": mode_val,
            "Max": round(series.max(), 2),
            "Min": round(series.min(), 2),
            "Range": round(series.max() - series.min(), 2),
            "Count": int(series.count()),
            "Std": round(series.std(), 2),
            "Variance": round(series.var(), 2),
            "Skewness": round(series.skew(), 2),
            "Kurtosis": round(series.kurtosis(), 2),
            "IQR": round(iqr, 2)
        }

    for col in numeric_cols:
        real_stats = stats_for_column(real_df, col)
        synth_stats = stats_for_column(synth_df, col)

        report_html += f"<h3>Column: {col}</h3>"
        report_html += """
        <table class="table table-bordered table-striped">
            <thead>
                <tr>
                    <th>Statistic</th>
                    <th>Real Data</th>
                    <th>Synthetic Data</th>
                </tr>
            </thead>
            <tbody>
        """
        for stat in real_stats.keys():
            report_html += f"""
                <tr>
                    <td>{stat}</td>
                    <td>{real_stats[stat]}</td>
                    <td>{synth_stats[stat]}</td>
                </tr>
            """
        report_html += "</tbody></table><br>"

    return report_html



# === Function 2: All Visualization Types ===
def generate_visualizations(real_df, synth_df):
    plot_paths = []
    os.makedirs('static/plots', exist_ok=True)

    real_df = real_df.fillna(real_df.mean(numeric_only=True))
    synth_df = synth_df.fillna(synth_df.mean(numeric_only=True))
    numeric_cols = real_df.select_dtypes(include='number').columns.tolist()

    # === UNIVARIATE PLOTS ===
    for col in numeric_cols[:4]:
        # Histogram
        plt.figure()
        sns.histplot(real_df[col], color='blue', kde=False, label='Real', stat='density')
        sns.histplot(synth_df[col], color='green', kde=False, label='Synthetic', stat='density')
        plt.legend()
        plt.title(f'Histogram - {col}')
        path = f'static/plots/histogram_{col}.png'
        plt.savefig(path); plt.close()
        plot_paths.append(path)

        # KDE
        plt.figure()
        sns.kdeplot(real_df[col], label='Real', fill=True)
        sns.kdeplot(synth_df[col], label='Synthetic', fill=True)
        plt.title(f'KDE Plot - {col}')
        plt.legend()
        path = f'static/plots/kde_{col}.png'
        plt.savefig(path); plt.close()
        plot_paths.append(path)

        # Boxplot
        plt.figure()
        sns.boxplot(data=[real_df[col], synth_df[col]], palette=["blue", "green"])
        plt.xticks([0, 1], ['Real', 'Synthetic'])
        plt.title(f'Boxplot - {col}')
        path = f'static/plots/boxplot_{col}.png'
        plt.savefig(path); plt.close()
        plot_paths.append(path)

        # Violin Plot
        plt.figure()
        sns.violinplot(data=[real_df[col], synth_df[col]], palette=["blue", "green"])
        plt.xticks([0, 1], ['Real', 'Synthetic'])
        plt.title(f'Violin Plot - {col}')
        path = f'static/plots/violin_{col}.png'
        plt.savefig(path); plt.close()
        plot_paths.append(path)

    # === BIVARIATE PLOTS ===
    if len(numeric_cols) >= 2:
        x, y = numeric_cols[0], numeric_cols[1]

        # Scatter Plot
        plt.figure()
        sns.scatterplot(x=real_df[x], y=real_df[y], color='blue', label='Real', alpha=0.6)
        sns.scatterplot(x=synth_df[x], y=synth_df[y], color='green', label='Synthetic', alpha=0.6)
        plt.legend()
        plt.title(f'Scatter Plot - {x} vs {y}')
        path = f'static/plots/scatter_{x}_{y}.png'
        plt.savefig(path); plt.close()
        plot_paths.append(path)

        # Joint KDE Plot
        g = sns.jointplot(data=real_df, x=x, y=y, kind='kde', fill=True, color='blue')
        g.fig.suptitle(f'Real KDE Joint Plot - {x} vs {y}')
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        path = f'static/plots/jointplot_real_{x}_{y}.png'
        g.fig.savefig(path); plt.close()
        plot_paths.append(path)

    # === MULTIVARIATE PLOTS ===
    if len(numeric_cols) >= 3:
        # Pairplot (Real)
        sns.pairplot(real_df[numeric_cols[:4]], diag_kind='kde', corner=True)
        plt.suptitle('Real Data Pairplot', y=1.02)
        path = 'static/plots/pairplot_real.png'
        plt.savefig(path); plt.close()
        plot_paths.append(path)

        # Correlation Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(real_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Real Data Correlation Heatmap')
        path = 'static/plots/heatmap_real.png'
        plt.savefig(path); plt.close()
        plot_paths.append(path)

    return plot_paths

from pathlib import Path

import pandas as pd
import plotly.express as px

# Give a result file
FILE_NAME = "TumDeneySonuclari.xlsx"
here = Path(__file__).resolve()
project_root = here.parents[2]
results_dir = project_root / "results" / "ml_pipeline_experiments" / "experimental_results" / FILE_NAME


def load_all_sheets(file_path):
    """
    Load each sheet in the Excel file, tag it by sheet name as Dataset,
    and return a concatenated DataFrame.
    """
    xls = pd.ExcelFile(file_path)
    df_list = []
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        df_list.append(df_sheet)

    df_res = pd.concat(df_list, ignore_index=True)
    return df_res


combined = load_all_sheets(results_dir)

# 3) Karşılaştırmak istediğiniz metrikler
metrics = ['Precision', 'Recall', 'F1', 'Accuracy']

# 4) Her metrik için ayrı bir violin çizimi
for m in metrics:
    fig = px.violin(
        combined,
        x='Aggregation',
        y=m,
        color='Dataset',
        box=True,  # kutu içi çeyreklikleri göster
        points='all',  # tüm noktaları serpiştir
        width=1200,
        height=600,
        labels={
            'Aggregation': 'Method',
            m: m,
            'Dataset': 'Dataset'
        }
    )
    # fig.update_traces(width=0.3)
    # Yazı boyutlarını büyütme
    fig.update_layout(
        title=f'Comparison of {m} across Methods and Datasets',
        violinmode='group',
        title_font=dict(size=22),
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
        xaxis_tickfont=dict(size=18),
        yaxis_tickfont=dict(size=18),
        legend_title_font=dict(size=18),
        legend_font=dict(size=16),
        margin=dict(l=60, r=20, t=60, b=60)
    )
    fig.for_each_trace(lambda t: t.update(showlegend=False))
    fig.show()

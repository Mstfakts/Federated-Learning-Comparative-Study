import pandas as pd
import plotly.graph_objects as go

data = {
    'Dataset': ['DCCC'] * 4 + ['HCDR'] * 4 + ['HMEQ'] * 4,
    'Metric': ['Precision', 'Recall', 'F1', 'Accuracy'] * 3,
    'Model_CML': [
        'LR+kBest', 'MLP+SMOTE', 'XGB+kBest+SMOTE', 'XGB+kBest',
        'XGB', 'XGB+PCA+RUS', 'XGB', 'SVM+PCA+SMOTE',
        'RF', 'XGB+kBest+RUS', 'RF+kBest+RUS', 'RF+SMOTE'
    ],
    'Score_CML': [0.73, 0.74, 0.53, 0.82, 0.27, 0.99, 0.31, 0.92, 0.79, 0.77, 0.77, 0.95],
    'Model_FedAvg': [
        'LR+kBest', 'LR+RUS', 'RF+RUS', 'MLP+kBest',
        'XGB+PCA', 'XGB', 'RF', 'XGB+PCA',
        'MLP', 'RF+RUS', 'RF+SMOTE', 'RF+SMOTE'
    ],
    'Score_FedAvg': [0.75, 0.74, 0.52, 0.82, 0.32, 0.88, 0.32, 0.92, 0.92, 0.88, 0.81, 0.93],
    'Model_FedF1': [
        'LR+kBest', 'LR+RUS', 'RF+kBest+SMOTE', 'MLP+kBest',
        'XGB+PCA', 'RF+PCA+RUS', 'XGB+kBest', 'XGB+kBest',
        'RF', 'RF+RUS', 'RF+SMOTE', 'RF+SMOTE'
    ],
    'Score_FedF1': [0.71, 0.75, 0.52, 0.82, 0.33, 0.69, 0.89, 0.92, 0.88, 0.87, 0.82, 0.93]
}
df_wide = pd.DataFrame(data)

df_list = []
approaches = {'CML': 'Centralized (CML)', 'FedAvg': 'Federated (FedAvg)', 'FedF1': 'Federated (FedF1)'}
for key, approach_name in approaches.items():
    temp_df = df_wide[['Dataset', 'Metric', f'Model_{key}', f'Score_{key}']].copy()
    temp_df.rename(columns={f'Model_{key}': 'Model', f'Score_{key}': 'Score'}, inplace=True)
    temp_df['Approach'] = approach_name
    df_list.append(temp_df)

df_long = pd.concat(df_list, ignore_index=True)


def create_benchmark_heatmap(dataset_name):
    df_dataset = df_long[df_long['Dataset'] == dataset_name]

    score_pivot = df_dataset.pivot_table(index='Metric', columns='Approach', values='Score')
    model_pivot = df_dataset.pivot_table(index='Metric', columns='Approach', values='Model', aggfunc='first')

    metric_order = ['Accuracy', 'F1', 'Recall', 'Precision']
    approach_order = ['Centralized (CML)', 'Federated (FedAvg)', 'Federated (FedF1)']
    score_pivot = score_pivot.reindex(index=metric_order, columns=approach_order)
    model_pivot = model_pivot.reindex(index=metric_order, columns=approach_order)

    fig = go.Figure(data=go.Heatmap(
        z=score_pivot.values,
        x=score_pivot.columns,
        y=score_pivot.index,
        text=model_pivot.values,
        texttemplate="<b>%{text}</b><br>%{z:.2f}",
        textfont={"size": 15},
        colorscale='Viridis',
        colorbar_title='Score',
        zmin=0,
        zmax=1
    ))

    fig.update_layout(
        title={
            'text': f'<b>Best Performance Results for {dataset_name}</b>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            side='top',
            title='Aggregation Methods'
        ),
        yaxis_title='Metrics',
        font=dict(size=12),
        height=500,
        width=900
    )

    fig.show()


for dataset in df_long['Dataset'].unique():
    create_benchmark_heatmap(dataset)

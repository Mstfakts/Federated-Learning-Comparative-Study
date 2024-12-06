import plotly.express as px
import pandas as pd

# Veriler
labels = ["Central ML", "FL with FedAvg", "FL with FedF1"]
central_ml = [0.71, 0.66, 0.44, 0.38, 0.67, 0.67, 0.61, 0.63, 0.39, 0.63, 0.73, 0.69, 0.68, 0.49, 0.67, 0.37, 0.53, 0.29, 0.29, 0.49, 0.37, 0.46, 0.42, 0.38, 0.50, 0.48, 0.54, 0.49, 0.49, 0.54, 0.37, 0.46, 0.40, 0.39, 0.47, 0.35, 0.45, 0.43, 0.39, 0.47, 0.48, 0.50, 0.48, 0.49, 0.50]
fedavg = [0.71, 0.68, 0.63, 0.37, 0.65, 0.70, 0.64, 0.58, 0.37, 0.66, 0.75, 0.68, 0.64, 0.49, 0.64, 0.39, 0.46, 0.37, 0.33, 0.53, 0.30, 0.34, 0.28, 0.31, 0.48, 0.47, 0.41, 0.45, 0.50, 0.55, 0.28, 0.49, 0.43, 0.38, 0.45, 0.30, 0.33, 0.29, 0.30, 0.44, 0.30, 0.47, 0.46, 0.47, 0.46]
fedf1 = [0.69, 0.66, 0.32, 0.37, 0.67, 0.67, 0.63, 0.59, 0.37, 0.63, 0.71, 0.69, 0.63, 0.50, 0.65, 0.39, 0.42, 0.29, 0.32, 0.52, 0.32, 0.35, 0.27, 0.33, 0.49, 0.46, 0.39, 0.45, 0.49, 0.54, 0.29, 0.48, 0.42, 0.38, 0.45, 0.29, 0.37, 0.28, 0.31, 0.46, 0.29, 0.48, 0.47, 0.50, 0.47]

# Verileri DataFrame'e donusturme
data = {
    "Precision": central_ml + fedavg + fedf1,
    "Method": [labels[0]] * len(central_ml) + [labels[1]] * len(fedavg) + [labels[2]] * len(fedf1)
}
df = pd.DataFrame(data)

# Violin grafiği oluşturma
fig = px.violin(df, x="Method", y="Precision", box=True, points="all",
                labels={"Method": "", "Precision": "Precision"},
                color="Method")

# Yazı boyutlarını büyütme
fig.update_layout(
    title_font=dict(size=20),
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24),
    legend_font=dict(size=20),
    xaxis_tickfont=dict(size=24),
    yaxis_tickfont=dict(size=24)
)

# Grafiği gösterme
fig.show()

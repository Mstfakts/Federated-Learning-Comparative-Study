import plotly.express as px
import pandas as pd

# Veriler
labels = ["Central ML", "FL with FedAvg", "FL with FedF1"]
central_ml = [0.23,0.36,0.19,0.62,0.37,0.20,0.33,0.38,0.61,0.34,0.21,0.36,0.34,0.54,0.36,0.59,0.53,0.74,0.70,0.56,0.64,0.59,0.62,0.62,0.55,0.54,0.53,0.57,0.55,0.52,0.67,0.64,0.65,0.64,0.62,0.67,0.62,0.64,0.60,0.59,0.54,0.57,0.59,0.53,0.57]
fedavg = [0.16,0.35,0.37,0.64,0.32,0.13,0.28,0.36,0.61,0.31,0.11,0.29,0.40,0.53,0.37,0.60,0.56,0.68,0.68,0.51,0.56,0.51,0.41,0.54,0.53,0.56,0.62,0.59,0.52,0.49,0.74,0.57,0.63,0.65,0.64,0.56,0.53,0.54,0.55,0.60,0.74,0.58,0.61,0.55,0.58]
fedf1 = [0.23,0.30,0.39,0.63,0.32,0.18,0.28,0.36,0.61,0.30,0.25,0.30,0.41,0.53,0.36,0.60,0.65,0.71,0.68,0.53,0.56,0.52,0.45,0.55,0.54,0.58,0.66,0.59,0.55,0.50,0.75,0.58,0.61,0.65,0.63,0.54,0.55,0.55,0.55,0.60,0.75,0.59,0.61,0.53,0.57]

# Verileri DataFrame'e donusturme
data = {
    "Recall": central_ml + fedavg + fedf1,
    "Method": [labels[0]] * len(central_ml) + [labels[1]] * len(fedavg) + [labels[2]] * len(fedf1)
}
df = pd.DataFrame(data)

# Violin grafiği oluşturma
fig = px.violin(df, x="Method", y="Recall", box=True, points="all",
                labels={"Method": "", "Recall": "Recall"},
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

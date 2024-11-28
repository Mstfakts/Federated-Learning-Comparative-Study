import plotly.express as px
import pandas as pd

# Veriler
labels = ["Central ML", "FL with FedAvg", "FL with FedF1"]
central_ml = [0.35,0.46,0.23,0.46,0.47,0.31,0.43,0.47,0.47,0.44,0.32,0.48,0.46,0.51,0.46,0.45,0.53,0.42,0.41,0.52,0.47,0.52,0.50,0.47,0.53,0.51,0.53,0.52,0.52,0.53,0.47,0.53,0.50,0.48,0.53,0.46,0.52,0.51,0.47,0.53,0.51,0.53,0.53,0.51,0.53]
fedavg = [0.26,0.46,0.46,0.46,0.42,0.22,0.39,0.43,0.45,0.42,0.18,0.39,0.49,0.50,0.46,0.47,0.50,0.47,0.44,0.52,0.39,0.40,0.33,0.39,0.50,0.50,0.49,0.50,0.50,0.51,0.40,0.52,0.50,0.47,0.52,0.38,0.41,0.37,0.39,0.50,0.39,0.51,0.52,0.50,0.51]
fedf1 = [0.34,0.40,0.34,0.46,0.42,0.29,0.38,0.44,0.46,0.40,0.37,0.40,0.49,0.51,0.46,0.47,0.50,0.40,0.43,0.52,0.40,0.42,0.33,0.41,0.51,0.50,0.47,0.51,0.51,0.52,0.41,0.52,0.49,0.47,0.52,0.38,0.44,0.36,0.39,0.52,0.39,0.52,0.52,0.50,0.51]

# Verileri DataFrame'e donusturme
data = {
    "F1": central_ml + fedavg + fedf1,
    "Method": [labels[0]] * len(central_ml) + [labels[1]] * len(fedavg) + [labels[2]] * len(fedf1)
}
df = pd.DataFrame(data)

# Violin grafiği oluşturma
fig = px.violin(df, x="Method", y="F1", box=True, points="all",
                labels={"Method": "", "F1": "F1"},
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

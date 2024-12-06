import plotly.express as px
import pandas as pd

# Veriler
labels = ["Central ML", "FL with FedAvg", "FL with FedF1"]
central_ml = [0.81,0.82,0.76,0.68,0.82,0.80,0.81,0.81,0.70,0.81,0.81,0.82,0.82,0.77,0.82,0.69,0.79,0.55,0.57,0.77,0.68,0.76,0.72,0.69,0.78,0.77,0.80,0.77,0.77,0.80,0.67,0.75,0.71,0.69,0.76,0.66,0.75,0.73,0.70,0.77,0.77,0.76,0.77,0.77,0.78]
fedavg = [0.80,0.82,0.81,0.68,0.81,0.79,0.81,0.80,0.69,0.82,0.79,0.81,0.82,0.77,0.82,0.71,0.76,0.68,0.62,0.79,0.62,0.66,0.64,0.64,0.77,0.76,0.71,0.75,0.78,0.80,0.53,0.76,0.73,0.69,0.75,0.62,0.65,0.60,0.62,0.75,0.50,0.76,0.76,0.77,0.78]
fedf1 = [0.81,0.81,0.68,0.68,0.81,0.80,0.81,0.80,0.69,0.81,0.81,0.81,0.82,0.78,0.82,0.70,0.71,0.55,0.62,0.79,0.63,0.68,0.61,0.64,0.78,0.74,0.67,0.75,0.77,0.80,0.53,0.76,0.73,0.69,0.75,0.61,0.68,0.58,0.64,0.76,0.48,0.76,0.76,0.78,0.77]

# Verileri DataFrame'e donusturme
data = {
    "Accuracy": central_ml + fedavg + fedf1,
    "Method": [labels[0]] * len(central_ml) + [labels[1]] * len(fedavg) + [labels[2]] * len(fedf1)
}
df = pd.DataFrame(data)

# Violin grafiği oluşturma
fig = px.violin(df, x="Method", y="Accuracy", box=True, points="all",
                labels={"Method": "", "Accuracy": "Accuracy"},
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

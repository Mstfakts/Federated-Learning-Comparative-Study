import plotly.graph_objects as go

# Verilen veriler
categories = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
pay_data = {
    'PAY_0': [-0.0167, 1.1238, -2, -1, 0, 0, 8],
    'PAY_2': [-0.1338, 1.1972, -2, -1, 0, 0, 8],
    'PAY_3': [-0.1662, 1.1969, -2, -1, 0, 0, 8],
    'PAY_4': [-0.2207, 1.1691, -2, -1, 0, 0, 8],
    'PAY_5': [-0.2662, 1.1332, -2, -1, 0, 0, 8],
    'PAY_6': [-0.2911, 1.1499, -2, -1, 0, 0, 8]
}

# PAY verilerinin grafikte gösterimi
fig = go.Figure()

for i, (pay, values) in enumerate(pay_data.items()):

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        name=pay,
        text=values,
        textposition='auto',  # Değerleri barların içine otomatik yerleştir
        textangle=-90,  # Değerleri 90 derece dikey yapmak
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=50)
    ))

fig.update_traces(textfont_size=50)  # Metin boyutunu ayarlamak için

# Grafiğin düzenlenmesi
fig.update_layout(
    yaxis_title='Values',
    yaxis=dict(range=[-3, 9]),
    barmode='group',
    legend_title='PAY Columns',
    font=dict(size=24)
)

fig.show()

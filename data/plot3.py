import plotly.graph_objects as go

# Verilen veriler
categories = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
pay_data = {
    'PAY_AMT1': [5663.58, 16563.28, 0, 1000  , 2100, 5006   , 8735 ],
    'PAY_AMT2': [5921.16, 23040.87, 0, 833   , 2009, 5000   , 16842],
    'PAY_AMT3': [5225.68, 17606.96, 0, 390   , 1800, 4505   , 8960 ],
    'PAY_AMT4': [4826.08, 15666.16, 0, 296   , 1500, 4013.25, 6210 ],
    'PAY_AMT5': [4799.39, 15278.31, 0, 252.5 , 1500, 4031.5 , 4265 ],
    'PAY_AMT6': [5215.50, 17777.47, 0, 117.75, 1500, 4000   , 5286 ]
}

# PAY verilerinin grafikte gösterimi
fig = go.Figure()

for i, (pay, values) in enumerate(pay_data.items()):
    text_values = values[:-1] + [str(values[-1]) + ' x 10^2']
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        name=pay,
        text=text_values,
        textposition='auto',  # Değerleri barların içine otomatik yerleştir
        textangle=-90,  # Değerleri 90 derece dikey yapmak
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=50)
    ))

fig.update_traces(textfont_size=50)  # Metin boyutunu ayarlamak için

# Grafiğin düzenlenmesi
fig.update_layout(
    yaxis_title='Values',
    #yaxis=dict(range=[-3, 9]),
    barmode='group',
    legend_title='PAY_AMT Columns',
    font=dict(size=24)
)

fig.show()

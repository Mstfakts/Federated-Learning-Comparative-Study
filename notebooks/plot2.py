import plotly.graph_objects as go

# Verilen veriler
categories = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
pay_data = {
    'BILL_AMT1': [51223.33, 73635.86, -165580, 3558.75, 22381.5 , 67091   , 96451 ],
    'BILL_AMT2': [49179.08, 71773.77, -69777 , 2984.75, 21200   , 64006.25, 98339 ],
    'BILL_AMT3': [47013.15, 69349.39, -157264, 2666.25, 20088.5 , 60164.75, 166408],
    'BILL_AMT4': [43262.95, 64332.86, -170000, 2326.75, 19052   , 54506   , 89158 ],
    'BILL_AMT5': [40311.40, 60797.16, -81334 , 1763   , 18104.5 , 50190.5 , 92711 ],
    'BILL_AMT6': [38871.76, 59554.11, -339603, 1256   , 17071   , 49198.25, 96166 ]
}

# PAY verilerinin grafikte gösterimi
fig = go.Figure()

for i, (pay, values) in enumerate(pay_data.items()):
    text_values = values[:-1] + [str(values[-1]) + 'x10']
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
    legend_title='BILL_AMT Columns',
    font=dict(size=24)
)

fig.show()

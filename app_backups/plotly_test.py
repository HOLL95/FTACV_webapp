from plotly.subplots import make_subplots
import plotly.graph_objects as go
import copy
num_harmonics=8
row_1=[[{"rowspan": num_harmonics}, {"rowspan": num_harmonics}, {"rowspan": num_harmonics}]]
num_cols=3
num_rows=3
empty_row=[None]*num_cols
[row_1.append(empty_row) for x in range(1, num_harmonics)]
specs=copy.deepcopy(row_1)
[specs.append(x) for x in row_1]
row_3=[[{}]*num_cols]*num_harmonics
[specs.append(x) for x in row_3]
harmonic_row=3

fig = make_subplots(
    rows=num_harmonics*num_rows, cols=num_cols,
    specs=specs,
    print_grid=True)
steps = []

for i in range(0, num_rows):
    row_val=(i*num_harmonics)+1
    if i!=(harmonic_row)-1:
        for j in range(1, num_cols+1):
            name=str(row_val)+","+str(j)
            print("row=", row_val, "col=", j)
            fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name=name), row=row_val, col=j)
    else:
        for j in range(1, num_cols+1):
            for q in range(0, num_harmonics):
                name=str(row_val)+","+str(j)
                print("row=", row_val, "col=", j)
                fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name=name), row=row_val+q, col=j)


fig.update_layout(height=1200,title_text="specs examples")
fig.show()

"""df = pd.DataFrame({'Machine': ['K2K01','K2K01','K2K01','K2K02','K2K02','K2K02','K2K03','K2K03','K2K03'],
                   'Units': [100,200,400,400,300,100,500,700,500],
                   'Time':[11,12,13,11,12,13,11,12,13]})

groups = df.groupby(by='Machine')

data = []
colors=['red', 'blue', 'green']

for group, dataframe in groups:
    dataframe = dataframe.sort_values(by=['Time'])
    trace = go.Scatter(x=dataframe.Time.tolist(),
                       y=dataframe.Units.tolist(),
                       marker=dict(color=colors[len(data)]),
                       name=group)
    data.append(trace)

layout =  go.Layout(xaxis={'title': 'Time'},
                    yaxis={'title': 'Produced Units'},
                    margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                    hovermode='closest')

figure = go.Figure(data=data, layout=layout)
figure.show()"""

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc

app = dash.Dash(__name__)
server=app.server
app.title = "Store problems"
app.layout=dbc.Container([dbc.Button("Apply", id="submit-button-state", color="primary", block=True, n_clicks=0),
            dcc.Store(id="stored_data", storage_type="local")])
@app.callback(
    Output("submit-button-state", "n_clicks"),
    [State("submit-button-state", "n_clicks")]
)
def store_num(clicks):
    if clicks>0:
        return 1

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)

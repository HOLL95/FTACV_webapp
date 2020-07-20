import dash_table
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
import numpy as np
import math
import copy # default='warn'
import matplotlib.pyplot as plt
import time
external_stylesheets = [dbc.themes.BOOTSTRAP]

RV_param_list={
    "E_0":-0.3,
    'E_start':  -500e-3, #(starting dc voltage - V)
    'E_reverse':100e-3,
    'omega':8.84,  #    (frequency Hz)
    "v":    22.35174e-3,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 10000, #(reaction rate s-1)
    'alpha': 0.5,
    "k0_scale":0.5,
    "k0_shape":0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "E0_skew":0,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :0,
}
table_params=RV_param_list.keys()
forbidden_params=["E0_mean", "E0_std", "E0_skew","k0_scale", "k0_shape",
                    "alpha_mean", "alpha_std"]

param_bounds={
    "E_start":[-2, 2],
    "E_reverse":[0, 4],
    "area":[1e-5, 0.1],
    "sampling_freq":[1/1000.0, 1/10.0],
    'E_0':[-2, 4],
    'omega':[1, 1e5],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 5e5],  #     (uncompensated resistance ohms)
    'Cdl': [0,2e-3], #(capacitance parameters)
    'CdlE1': [-0.01,0.01],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [1e-4*RV_param_list["original_gamma"],1e4*RV_param_list["original_gamma"]],
    'k_0': [0.1, 1e6], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0, 2*math.pi],
    "E0_mean":[-2,4],
    "E0_std": [1e-4,  1],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0, 2*math.pi],
}
time_start=2/(RV_param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":16,
    "test": False,
    "method": "ramped",
    "likelihood":"timeseries",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[]
}
table_dict={key:RV_param_list[key] for key in param_bounds.keys()}
for param in forbidden_params:
    del table_dict[param]
table_data=[{"Parameter":key, "Value":table_dict[key]} for key in table_dict.keys()]
table_names=list(table_dict.keys())
SV_simulation_options=copy.deepcopy(simulation_options)
SV_param_list=copy.deepcopy(RV_param_list)
changed_SV_params=["d_E", "phase", "cap_phase", "num_peaks", "original_omega"]
changed_sv_vals=[300e-3, 3*math.pi/2,  3*math.pi/2, 50, RV_param_list["omega"]]
for key, value in zip(changed_SV_params, changed_sv_vals):
    SV_param_list[key]=value
num_harms=6
start_harm=1
SV_simulation_options["method"]="sinusoidal"
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(start_harm,start_harm+num_harms,1)),
    "bounds_val":20000,
}
RV=single_electron(None, RV_param_list, simulation_options, other_values, param_bounds)
SV=single_electron(None, SV_param_list, SV_simulation_options, other_values, param_bounds)
RV.def_optim_list(list(table_names))

SV.def_optim_list([])

SV_simulation_options["no_transient"]=2/SV_param_list["omega"]
SV_new=single_electron(None, SV_param_list, SV_simulation_options, other_values, param_bounds)

timeseries=SV_new.test_vals([], "timeseries")
SV_plot_times=SV_new.t_nondim(SV_new.time_vec[SV_new.time_idx])
SV_plot_voltages=SV_new.e_nondim(SV_new.define_voltages()[SV_new.time_idx])

plot_height=max(75*num_harms, 450)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "FTACVSIM"
parameter_names=RV_param_list.keys()
parameter_sliders=[]
forbidden_params=["v", "d_E"]
for key in parameter_names:
    if key in param_bounds:
        parameter_sliders.append(html.Div([
                # Create element to hide/show, in this case an 'Input Component'
                dbc.Label(key,id=key+"_slider_id", hidden=True),
                dbc.Input(
                id = key+"_slider",
                placeholder = key,
                type="number",
                min=param_bounds[key][0], max=param_bounds[key][1], step=abs(param_bounds[key][1]-param_bounds[key][0])/1000
                ),
            ], style= {'display': 'block'}
            ),
        )
controls = dbc.Card([dbc.FormGroup(

                                [

                                dbc.Label("Parameters"),
                                dcc.Dropdown(
                                                id='parameter_list',
                                                options=[{'label': x, 'value': x} for x in table_names if x not in forbidden_params],
                                                multi=True,
                                            ),
                                            ]
                                ),
                                *parameter_sliders,
                                dbc.FormGroup(
                                    [
                                        dbc.Label('Parameter values'),
                                        dash_table.DataTable(
                                            id='param_table',
                                            columns=[
                                                {"name": "Parameter", "id": "Parameter"},
                                                {"name": "Value", "id": "Value"},
                                            ],
                                            data=[
                                                {"Parameter":key, "Value":table_dict[key]} for key in table_dict.keys()
                                            ],
                                            style_cell_conditional=[
                                                {'if': {'column_id': 'Parameter'},
                                                 'width': '5px'},
                                                {'if': {'column_id': 'Value'},
                                                 'width': '10px'},
                                            ],
                                            style_cell={'textAlign': 'left',
                                                        'fontSize': 16, 'font-family': 'Helvetica'},
                                            style_header={
                                                'backgroundColor': 'white',
                                                'fontWeight': 'bold'
                                            },

                                        ),
                                    ]
                                ),
                                dbc.Button("Apply", id="submit-button-state",
                                           color="primary", block=True, n_clicks=0)

                               ],body=True)



app.layout = dbc.Container(
    [
        dbc.Jumbotron(
            [
                dbc.Container(
                    [
                        html.H1("FTACVSIM", className="display-3"),
                        html.P(
                            "Interactively simulate different Coronavirus scenarios. ",
                            className="lead",
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''
                            NOTHING.
                            '''
                                     )
                    ],
                    fluid=True,
                )
            ],
            fluid=True,
            className="jumbotron bg-white text-dark"
        ),
        dbc.Col([dbc.Row([
        dbc.ListGroup([dbc.ListGroupItemHeading("Ramped plot type"),
        dcc.RadioItems(
        options=[
            {'label': 'Current-Voltage ', 'value': 'r_x_volt',},
            {'label': 'Current-Time ', 'value': 'r_x_time',},
            {'label': 'Voltage-Time ', 'value': 'r_x_volt_time',},
            {'label': 'Fourier spectrum (abs) ', 'value': 'r_x_fft',},
            {'label': 'Freeze plot ', 'value': 'r_x_freeze',},
        ],
        id="r_button",
        value='r_x_time',
        labelStyle={'display': 'inline-block'},
        style={"margin-right": "5px"})],style={"margin-right": "15px"}),]),

        dbc.Row([dbc.ListGroup([dbc.ListGroupItemHeading("Sinusoidal plot type"),
        dcc.RadioItems(
        options=[
            {'label': 'Current-Voltage ', 'value': 's_x_volt',},
            {'label': 'Current-Time ', 'value': 's_x_time',},
            {'label': 'Voltage-Time ', 'value': 's_x_volt_time',},
            {'label': 'Fourier spectrum (real) ', 'value': 's_x_fft',},
            {'label': 'Freeze plot ', 'value': 's_x_freeze',},
        ],
        id="s_button",
        value='s_x_volt',
        labelStyle={'display': 'inline-block'},
        style={"margin-right": "5px"})],style={"margin-right": "15px"}),]),
        dbc.Row([dbc.ListGroup([dbc.ListGroupItemHeading("DC plot options", style={"margin-right": "5px"}),
        dcc.RadioItems(
        options=[
            {'label': 'Current-Voltage ', 'value': 'd_x_volt',},
            {'label': 'Current-Time ', 'value': 'd_x_time',},
            {'label': 'Voltage-Time ', 'value': 'd_x_volt_time',},
            {'label': 'Freeze plot ', 'value': 'd_x_freeze',},
        ],
        value='d_x_volt',
        id="d_button",
        labelStyle={'display': 'inline-block'},
        style={"margin-right": "5px"}),],style={"margin-right": "15px"})]),]),


        dbc.Row(
                [
                # here we place the controls we just defined,
                # and tell them to use up the left 3/12ths of the page.
                dbc.Col(controls, md=2),
                # now we place the graphs on the page, taking up
                # the right 9/12ths.
                dbc.Col([dcc.Graph(id='ramped_graph',figure=go.Figure(layout=dict(height=plot_height))),
                        dcc.Graph(id='sv_graph',figure=go.Figure(layout=dict(height=plot_height))),
                        dcc.Graph(id='DCV_graph',figure=go.Figure(layout=dict(height=plot_height))),],md=5),
                dbc.Col(
                        [dcc.Graph(id="ramped_harm_"+str(x),figure=go.Figure(layout=dict(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0}))) for x in range(start_harm, start_harm+num_harms)]+
                        [dcc.Graph(id="sv_harm_"+str(x),figure=go.Figure(layout=dict(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0}))) for x in range(start_harm, start_harm+num_harms)]
                        ,md=5),],align="top"
                ),
    ],
    # fluid is set to true so that the page reacts nicely to different sizes etc.
    fluid=True,
)
def plot_args(plot_class, option, params, method):
    if option!=(method+"_x_volt_time"):
        timeseries=plot_class.i_nondim(plot_class.test_vals(params, "timeseries"))
    else:
        plot_class.update_params(params)
        timeseries=[]
    times=plot_class.t_nondim(plot_class.time_vec[plot_class.time_idx])
    volts=plot_class.e_nondim(plot_class.define_voltages()[plot_class.time_idx])
    if option==(method+"_x_volt"):
        y_arg=timeseries
        x_arg=volts
    if options==(method+"_x_time"):
        y_arg=timeseries
        x_arg=time
    if option==(method+"_x_volt_time"):
        y_arg=volts
        x_arg=time
    if option==(method+"_x_fft"):
        y_arg=np.fft.fft(timeseries)[:(len(timeseries)/2)+1]
        x_arg=np.fft.fftfreq(len(timeseries), times[1]-times[0])[:(len(timeseries)/2)+1]
    if method=="r":
        return x_arg, y_arg, timeseries, times
    elif method=="s":
        return x_arg, y_arg, timeseries, times, volts

def show_hide_element(slider_name, drop_down_id):
    if drop_down_id is None:
        return {"display":"none"}, True
    for parameter in drop_down_id:
        if parameter+"_slider"==slider_name:
            return {"display":"block"}, False
    return {"display":"none"}, True
for key in parameter_names:
    if key in param_bounds:
        app.callback(
           [Output(component_id=key+"_slider", component_property='style'),
           Output(component_id=key+"_slider_id", component_property='hidden')],
           [Input(component_id=key+"_slider", component_property='id'),
           Input(component_id="parameter_list", component_property="value")],)(show_hide_element)

ramped_plots=[Output('ramped_graph', 'figure')]+[Output('ramped_harm_'+str(x), 'figure') for x in range(start_harm, start_harm+num_harms)]
sv_plots=[Output('sv_graph', 'figure')]+[Output('sv_harm_'+str(x), 'figure') for x in range(start_harm, start_harm+num_harms)]
tables=[Output("param_table", "data")]
harmonic_output=ramped_plots+sv_plots+tables
@app.callback(
                harmonic_output,
                [Input('submit-button-state', 'n_clicks')],
                [State(x+"_slider", "value") for x in table_names]+
                [State("param_table", "data")]+
                [State(component_id="parameter_list", component_property="value")]+
                [State(x+"_button", "value") for x in ["r", "s", "d"]]
                )
def apply_slider_changes(n_clicks, *inputs):
    slider_input_len=len(table_names)
    states=inputs[:slider_input_len]
    table_data=inputs[slider_input_len]
    drop_down_opts=inputs[slider_input_len+1]
    r_button, s_button, d_button=inputs[slider_input_len+2:]

    for i in range(0, len(states)):
        if states[i] is not None:
            val=states[i]
            val=min(val, param_bounds[table_names[i]][1])
            val=max(val, param_bounds[table_names[i]][0])
            RV.dim_dict[RV.optim_list[i]]=val
            SV.dim_dict[RV.optim_list[i]]=val
    SV.dim_dict["original_gamma"]=SV.dim_dict["gamma"]
    RV.dim_dict["original_gamma"]=RV.dim_dict["gamma"]
    SV.dim_dict["original_omega"]=SV.dim_dict["omega"]
    SV.simulation_options["no_transient"]=2/SV.dim_dict["omega"]
    RV.simulation_options["no_transient"]=2/RV.dim_dict["omega"]
    SV.simulation_options["optim_list"]=[]
    RV.simulation_options["optim_list"]=[]
    SV.dim_dict["d_E"]=(SV.dim_dict["E_reverse"]-SV.dim_dict["E_start"])/2
    RV.dim_dict["d_E"]=(RV.dim_dict["E_reverse"]-RV.dim_dict["E_start"])/4
    RV_new=single_electron(None, RV.dim_dict, RV.simulation_options, RV.other_values, param_bounds)
    SV_new=single_electron(None, SV.dim_dict, SV.simulation_options, SV.other_values, param_bounds)

    params=[]
    ramped_layout=go.Layout(height=plot_height)
    harmonic_layout=go.Layout(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0})
    if n_clicks>0:
        for q in range(0, len(table_data)):
            table_data[q]["Value"]=RV_new.dim_dict[table_data[q]["Parameter"]]


        ramped_x_val, ramped_y_val, r_timeseries, r_times=plot_args(RV_new, r_button, params, "r")
        sinusoidal_x_val, sinusoidal_y_val, s_timeseries, s_times, s_volts=plot_args(SV_new, s_button, params, "s")
        #SPACEFORDCV
        if r_button!="r_x_volt_time":
            r_harms=harmonics(list(range(start_harm, start_harm+num_harms)), RV_new.dim_dict["omega"], 0.05)
            ramped_harmonics=r_harms.generate_harmonics(r_times[0::10], r_timeseries[0::10], hanning=True)
        else:
            ramped_harmonics=np.zeros((num_harms, len(r_timeseries)))
        if s_button!="s_x_volt_time":
            s_harms=harmonics(list(range(start_harm, start_harm+num_harms)), SV_new.dim_dict["omega"], 0.05)
            SV_harmonics=s_harms.generate_harmonics(s_times, s_timeseries, hanning=False)
        else:
            SV_harmonics=np.zeros((num_harms, len(s_timeseries)))


        return_arg=[{"data": [
            {"x":ramped_x_val, "y": ramped_y_val, "type": "line", "name": "Ramped", "render_mode":"webgl"},
        ],
        "layout": ramped_layout,
        }]
        for i in range(0, num_harms):
            return_arg.append({"data": [
                {"x":r_times, "y": np.abs(ramped_harmonics[i,:]), "type": "line", "name": "Ramped_harm"+str(i), "render_mode":"webgl"},
            ],
            "layout": harmonic_layout,
            })
        return_arg.append({"data": [
            {"x":sinusoidal_x_val, "y": sinusoidal_y_val, "type": "line", "name": "SV", "render_mode":"webgl"},
        ],
        "layout": ramped_layout,
        })

        for i in range(0, num_harms):
            return_arg.append({"data": [
                {"x":s_volts, "y": np.real(SV_harmonics[i,:]), "type": "line", "name": "sv_harm"+str(i), "render_mode":"webgl"},
            ],
            "layout": harmonic_layout,
            })
        return_arg.append(table_data)
        return return_arg


    return [{"layout": ramped_layout}]+[{"layout": harmonic_layout}]*(num_harms)+[{"layout": ramped_layout}]+[{"layout": harmonic_layout}]*(num_harms)+[table_data]



if __name__ == '__main__':
    app.run_server(debug=True)

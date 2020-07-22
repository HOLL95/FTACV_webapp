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
from scipy.signal import decimate
from scipy.integrate import odeint
import math
import copy
import time
import rdp_lines
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
    'Cdl': 1e-4, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 10000, #(reaction rate s-1)
    'alpha': 0.5,
    "k0_scale":100,
    "k0_shape":0.5,
    "E0_mean":-0.3,
    "E0_std": 0.01,
    "E0_skew":0,
    "cap_phase":4.712388,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :4.712388,
}
table_params=RV_param_list.keys()
disped_params=["E0_mean", "E0_std", "E0_skew","k0_scale", "k0_shape",
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
orig_table_dict=copy.deepcopy(table_dict)
#for param in forbidden_params:
#    del table_dict[param]
table_data=[{"Parameter":key, "Value":table_dict[key]} for key in table_dict.keys()]
table_names=list(table_dict.keys())
SV_simulation_options=copy.deepcopy(simulation_options)
DCV_simulation_options=copy.deepcopy(simulation_options)
DCV_simulation_options["method"]="dcv"
SV_param_list=copy.deepcopy(RV_param_list)
changed_SV_params=["d_E", "phase", "cap_phase", "num_peaks", "original_omega", "sampling_freq"]
changed_sv_vals=[300e-3, 3*math.pi/2,  3*math.pi/2, 30, RV_param_list["omega"], 1/500.0]
for key, value in zip(changed_SV_params, changed_sv_vals):
    SV_param_list[key]=value
num_harms=7
start_harm=1
SV_simulation_options["method"]="sinusoidal"
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(start_harm,start_harm+num_harms,1)),
    "bounds_val":20000,
}
RV=single_electron(None, RV_param_list, simulation_options, other_values, param_bounds)
SV=single_electron(None, SV_param_list, SV_simulation_options, other_values, param_bounds)
DCV=single_electron(None, RV_param_list, DCV_simulation_options, other_values, param_bounds)
RV.def_optim_list(list(table_names))

SV.def_optim_list([])
DCV.def_optim_list([])

SV_simulation_options["no_transient"]=2/SV_param_list["omega"]
SV_new=single_electron(None, SV_param_list, SV_simulation_options, other_values, param_bounds)

timeseries=SV_new.test_vals([], "timeseries")
SV_plot_times=SV_new.t_nondim(SV_new.time_vec[SV_new.time_idx])
SV_plot_voltages=SV_new.e_nondim(SV_new.define_voltages()[SV_new.time_idx])

plot_height=max(100*num_harms, 600)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server
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
                min=param_bounds[key][0], max=param_bounds[key][1], step=min(1,abs(param_bounds[key][1]-param_bounds[key][0])/1000)
                ),
            ], style= {'display': 'block'}
            ),
        )

table_init=[{"Parameter":key, "Value":table_dict[key]} for key in table_dict.keys() if key not in disped_params]
print(table_init)
controls = dbc.Card([dbc.FormGroup([


                                dbc.ListGroup([dbc.ListGroupItemHeading("Dispersion bins"),dcc.Slider(
                                    min=1,
                                    max=26,
                                    step=1,
                                    marks={
                                    key:value for key,value in zip(range(1, 26, 5), [str(x) for x in range(1, 26, 5)])

                                    },
                                    value=16,
                                    id="dispersion_slider"
                                ),],id="slider_group",style= {'display': 'none'}),
                                dbc.ListGroup([dbc.ListGroupItemHeading("Parameters"),
                                dcc.Dropdown(
                                                id='parameter_list',
                                                options=[{'label': x, 'value': x} for x in table_names if key not in forbidden_params],
                                                multi=True,
                                            ),
                                            ]
                                ),
                                *parameter_sliders]),


                                dbc.FormGroup(
                                    [
                                        dbc.ListGroup([dbc.ListGroupItemHeading('Parameter values'),
                                        dash_table.DataTable(
                                            id='param_table',
                                            columns=[
                                                {"name": "Parameter", "id": "Parameter"},
                                                {"name": "Value", "id": "Value"},
                                            ],
                                            data=table_init,
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
                                ),]),
                                dbc.Button("Apply", id="submit-button-state",
                                           color="primary", block=True, n_clicks=0),
                                dbc.Button("Save plots", id="save_plot_state",
                                          color="primary", block=True, n_clicks=0),
                                dbc.Button("Reset parameters", id="reset_params",
                                       color="danger", block=True, n_clicks=0),
                                dbc.Button("Clear saved plots", id="reset_plots",
                                           color="danger", block=True, n_clicks=0)


                               ],body=True)


ramped_harm_init=[dcc.Graph(id="ramped_harm_"+str(x),figure=go.Figure(layout=dict(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0}))) for x in range(start_harm, start_harm+num_harms)]
sinusoidal_harm_init=[dcc.Graph(id="sinusoidal_harm_"+str(x),figure=go.Figure(layout=dict(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0}))) for x in range(start_harm, start_harm+num_harms)]
history_data={"ramped_plots":{}, "sinusoidal_plots":{}, "dcv_plots":{},"ramped_harmonics":{}, "sinusoidal_harmonics":{},
                "counters":{"ramped":0, "dcv":0, "sinusoidal":0}}
app.layout = dbc.Container(
    [
        dbc.Jumbotron(
            [
                dbc.Container(
                    [
                        html.H1("FTACVSIM", className="display-3"),
                        html.P(
                            "Interactively simulate different electrochemical parameters and the effects on three electrochemical experiments - ramped Fourier transform AC voltammetry,sinusoidal voltammetry and DC voltammetry. ",
                            className="lead",
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''
                            * To keep the results of the simulation, click the "Save plots" button. A new plot will be added, which is a copy of the current simulation. The results of up to two simulations can be saved in this manner.
                            * If you are accessing this simulation program online, heroku sets a limit of 30 seconds before timing out. If this occurs, nothing will happen after you click apply (i.e. the name of the browser tab will switch
                            from "Updating" to "FTACVSIM" without any changes occuring). If this happens, try only simualting one or two experiments (the ramped experiment is by far the slowest to simulate)
                            * Certain combinations of parameters, particularly high Ru and CdlE1-3 values will cause the static solver to crash. If this happens try the adaptive solver. However, this is very slow,
                            so refer to the above bullet point.
                            * There are three options for ramped plotting. The default decimates the data, which causes sampling artefacts. "Better ramped plots" removes the decimation, and the experimental "Better ramped plots"
                            uses a range of algorithms to reduced the number of points, with the introduction of some distortion.
                            * If timing out is becoming a serious issue, please refer to the [github](https://github.com/HOLL95/FTACV_webapp), and attempt to run the program sessionly.

                            '''
                                     )
                    ],
                    fluid=True,
                )
            ],
            fluid=True,
            className="jumbotron bg-white text-dark"
        ),
        dbc.Row([
        dbc.Col(dbc.ListGroup([dbc.ListGroupItemHeading("ramped plot options"),dcc.RadioItems(
            options=[
                {"label":"Decimated ramped plots", "value":"decimation"},
                {'label': 'Better ramped plots (slow tab switching)', 'value': 'no_decimation'},
                {'label': 'Better ramped plots (fast tab switching - experimental)', 'value': 'ramped_rdp'},
            ],
            value="decimation",
            id="plot_buttons",
        )])) ,
        dbc.Col(dbc.ListGroup([dbc.ListGroupItemHeading("Simulation experiments"),dcc.Checklist(
            options=[
                {'label': 'ramped', 'value': 'r_freeze'},
                {'label': 'Sinusoidal', 'value': 's_freeze'},
                {'label': 'DC', 'value': 'd_freeze'}
            ],
            value=["r_freeze", "s_freeze", "d_freeze"],
            id="freeze_buttons",
        )])) ,
        dbc.Col(dbc.ListGroup([dbc.ListGroupItemHeading("Adaptive simulation"),
        dcc.Checklist(
            options=[
                {'label': 'ramped', 'value': 'r_scipy'},
                {'label': 'Sinusoidal', 'value': 's_scipy'},
                {'label': 'DC', 'value': 'd_scipy'}
            ],
            value=[],
            id="adaptive_buttons",
        )])) ,]),
        dbc.Row(
                [
                # here we place the controls we just defined,
                # and tell them to use up the left 3/12ths of the page.
                dbc.Col(controls, md=2),
                # now we place the graphs on the page, taking up
                # the right 9/12ths.
                #dcc.Graph(id='ramped_graph',figure=go.Figure(layout=dict(height=plot_height)))
                #dcc.Graph(id='sv_graph',figure=go.Figure(layout=dict(height=plot_height))),
                dbc.Col([dcc.Store(storage_type="session", id="ramped_store", ),dcc.Store(storage_type="session", id="history_store", data=history_data),dcc.Store(storage_type="session", id="harmonic_store"),
                    dbc.Tabs(
                            [
                                dbc.Tab(label="Current-time", tab_id="r_time"),
                                dbc.Tab(label="Current-voltage", tab_id="r_volt"),
                                dbc.Tab(label="Voltage-time", tab_id="r_volt_time"),
                                dbc.Tab(label="Absolute FFT", tab_id="r_fft"),

                            ],
                            id="ramped_tabs",
                            active_tab="r_time",
                        ),
                        html.Div(id="ramped-tab-content", className="p-4"),
                        dcc.Store(storage_type="session", id="sinusoid_store"),
                        dbc.Tabs(
                                [
                                    dbc.Tab(label="Current-time", tab_id="s_time"),
                                    dbc.Tab(label="Current-voltage", tab_id="s_volt"),
                                    dbc.Tab(label="Voltage-time", tab_id="s_volt_time"),
                                    dbc.Tab(label="Real FFT", tab_id="s_r_fft"),
                                ],
                                id="sinusoidal_tabs",
                                active_tab="s_volt",
                            ),
                        html.Div(id="sinusoidal-tab-content", className="p-4"),
                        dcc.Store(storage_type="session", id="dcv_store"),
                        dbc.Tabs(
                                [
                                    dbc.Tab(label="Current-time", tab_id="d_time"),
                                    dbc.Tab(label="Current-voltage", tab_id="d_volt"),
                                    dbc.Tab(label="Voltage-time", tab_id="d_volt_time"),
                                ],
                                id="dc_tabs",
                                active_tab="d_time",
                            ),
                        html.Div(id="dc-tab-content", className="p-4"),],md=5),
                dbc.Col(
                        [html.Div(ramped_harm_init),
                        html.Div(sinusoidal_harm_init, style={"margin-top":"75px"})]
                        ,md=5),],align="top"
                ),
    ],
    fluid=True,
)
def running_reduction(series, N):
    running_mean=np.convolve(series, np.ones((N,))/N, mode='valid')
    return running_mean[0::N]
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
@app.callback(
    [Output("history_store", "data"), Output("reset_plots", "n_clicks")],
    [Input("save_plot_state", "n_clicks")],
    [State("ramped_store", "data"),
    State("sinusoid_store", "data"),
    State("dcv_store", "data"),
    State("harmonic_store", "data"),
    State("reset_plots", "n_clicks"),
    State("submit-button-state", "n_clicks"),
    State("history_store", "data")]
)
def move_data_to_hidden(n_clicks, ramped_data, sinusoid_data, dcv_data, harmonic_store, *stores):
    #start=time.time()
    max_plots=2
    current_history_labels=["ramped", "sinusoidal", "dcv"]
    if n_clicks>0 and stores[-2]>0:
        left_plots=dict(zip(current_history_labels, [ramped_data, sinusoid_data, dcv_data]))
        current_history=stores[-1]
        for label in current_history_labels:
            if left_plots[label] is not None:
                current_history["counters"][label]+=1
                plot_number=current_history["counters"][label]%max_plots
                current_history[label+"_plots"][label+"_"+str(plot_number)]=left_plots[label]
            if label is not "dcv":
                print(harmonic_store.keys())
                if harmonic_store[label] is not None:
                    plot_number=current_history["counters"][label]%max_plots
                    current_history[label+"_harmonics"][label+"_"+str(plot_number)]=harmonic_store[label]
        #print("storage_time", time.time()-start)
        return [current_history,0]
    return [history_data,0]
@app.callback(
    Output("slider_group", "style"),
    [Input("parameter_list", "value")]
)
def show_hide_slider(drop_down_ids):
    dispersion=False
    if drop_down_ids is None:
        return {"display":"none"}
    for param in drop_down_ids:
        if param in disped_params:
            dispersion=True
            break
    if dispersion==True:
        return {"display":"block"}
    else:
        return {"display":"none"}
@app.callback(
    Output("ramped-tab-content", "children"),
    [Input("ramped_tabs", "active_tab") , Input("ramped_store", "data"), Input("history_store", "modified_timestamp"), Input("reset_plots", "n_clicks")],
    [State("history_store", "data")]
)
def render_ramped_tab(active_tab, data, time, reset_button, existing_data):
    if active_tab and data is not None:
        past_plots=existing_data["ramped_plots"].keys()
        if len(past_plots)>0 and reset_button==0:
            for key in past_plots:
                existing_data["ramped_plots"][key][active_tab]["data"][0]["name"]=key
                data[active_tab]["data"].append(existing_data["ramped_plots"][key][active_tab]["data"][0])
            #data[active_tab]["data"]=apply_data
            return dcc.Graph(figure=data[active_tab])
        else:
            return  dcc.Graph(figure=data[active_tab])#layout=dict(height=plot_height))
    return "Click on a tab to view"
@app.callback(
    Output("sinusoidal-tab-content", "children"),
    [Input("sinusoidal_tabs", "active_tab"), Input("sinusoid_store", "data"), Input("history_store", "modified_timestamp"),Input("reset_plots", "n_clicks")],
    [State("history_store", "data")]
)
def render_sinusoid_tab(active_tab, data, time, reset_button, existing_data):
    if active_tab and data is not None:
        past_plots=existing_data["sinusoidal_plots"].keys()
        if len(past_plots)>0 and reset_button==0:
            for key in past_plots:
                existing_data["sinusoidal_plots"][key][active_tab]["data"][0]["name"]=key
                data[active_tab]["data"].append(existing_data["sinusoidal_plots"][key][active_tab]["data"][0])
            #data[active_tab]["data"]=apply_data
            return dcc.Graph(figure=data[active_tab])
        else:
            return  dcc.Graph(figure=data[active_tab])#layout=dict(height=plot_height))
    return "Click on a tab to view"
@app.callback(
    Output("dc-tab-content", "children"),
    [Input("dc_tabs", "active_tab"), Input("dcv_store", "data"), Input("history_store", "modified_timestamp"),Input("reset_plots", "n_clicks")],
    [State("history_store", "data")]
)
def render_dc_tab(active_tab, data, time, reset_button, existing_data):
    if active_tab and data is not None:
        past_plots=existing_data["dcv_plots"].keys()
        if len(past_plots)>0 and reset_button==0:
            for key in past_plots:
                existing_data["dcv_plots"][key][active_tab]["data"][0]["name"]=key
                data[active_tab]["data"].append(existing_data["dcv_plots"][key][active_tab]["data"][0])
            #data[active_tab]["data"]=apply_data
            return dcc.Graph(figure=data[active_tab])
        else:
            return  dcc.Graph(figure=data[active_tab])#layout=dict(height=plot_height))
    return "Click on a tab to view"
@app.callback(
    [Output(x+"_slider", "value") for x in table_names],
    [Input('reset_params', 'n_clicks')]
)
def reset_parameters(button_click):

    slider_vals=[orig_table_dict[x] for x in orig_table_dict.keys()]
    return slider_vals

ramped_plots=[Output('ramped_store', 'data')]#
sv_plots=[Output('sinusoid_store', 'data')]#[Output('sinusoidal_harm_'+str(x), 'figure') for x in range(start_harm, start_harm+num_harms)]
DCV_plots=[Output('dcv_store', 'data')]
sinusoidal_plots=[Output('harmonic_store', 'data')]
tables=[Output("param_table", "data")]
harmonic_output=ramped_plots+sv_plots+DCV_plots+sinusoidal_plots+tables
@app.callback(
                harmonic_output,
                [Input('submit-button-state', 'n_clicks')],
                [State(x+"_slider", "value") for x in table_names]+
                [State("param_table", "data")]+
                [State(component_id="parameter_list", component_property="value")]+
                [State("dispersion_slider", "value")]+
                [State("freeze_buttons", "value")]+
                [State("adaptive_buttons", "value")]+
                [State("plot_buttons", "value")]
                )
def apply_slider_changes(n_clicks, *inputs):
    #start=time.time()
    slider_input_len=len(table_names)
    states=inputs[:slider_input_len]
    table_data=inputs[slider_input_len]
    drop_down_opts=inputs[slider_input_len+1]
    disp_bins=inputs[slider_input_len+2]
    freeze_buttons=inputs[slider_input_len+3]
    adaptive_buttons=inputs[slider_input_len+4]
    plot_buttons=inputs[slider_input_len+5]
    #print(disp_bins, freeze_buttons, adaptive_buttons)
    dispersion_optim_list=[]
    if drop_down_opts is not None:
        dispersion_groups={"E_0":["E0_mean", "E0_std"], "k_0":["k0_shape", "k0_scale"], "alpha":["alpha_mean", "alpha_std"]}
        dispersion_associations={"E0_mean":"E_0", "E0_std":"E_0", "E0_skew":"E_0", "k0_shape":"k_0", "k0_scale":"k_0", "alpha_mean":"alpha", "alpha_std":"alpha"}
        dispersed_params=list(set([dispersion_associations[key] for key in drop_down_opts if key in dispersion_associations.keys()]))
        if len(dispersed_params)!=0:
            dispersion=True
            for key in dispersed_params:
                dispersion_optim_list+=dispersion_groups[key]
        else:
            dispersion=False
            params=[]
    else:
        dispersion=False
        params=[]
    for i in range(0, len(states)):
        if states[i] is not None:
            val=states[i]
            val=min(val, param_bounds[table_names[i]][1])
            val=max(val, param_bounds[table_names[i]][0])
            RV.dim_dict[RV.optim_list[i]]=val
            SV.dim_dict[RV.optim_list[i]]=val
            DCV.dim_dict[RV.optim_list[i]]=val
    SV.dim_dict["original_gamma"]=SV.dim_dict["gamma"]
    RV.dim_dict["original_gamma"]=RV.dim_dict["gamma"]
    DCV.dim_dict["original_gamma"]=DCV.dim_dict["gamma"]
    SV.dim_dict["original_omega"]=SV.dim_dict["omega"]
    SV.simulation_options["no_transient"]=2/SV.dim_dict["omega"]
    RV.simulation_options["no_transient"]=2/RV.dim_dict["omega"]
    DCV.simulation_options["no_transient"]=False
    SV.simulation_options["optim_list"]=[]
    RV.simulation_options["optim_list"]=[]
    DCV.simulation_options["optim_list"]=[]
    SV.dim_dict["d_E"]=(SV.dim_dict["E_reverse"]-SV.dim_dict["E_start"])/2
    RV.dim_dict["d_E"]=(RV.dim_dict["E_reverse"]-RV.dim_dict["E_start"])/4
    RV_new=single_electron(None, RV.dim_dict, RV.simulation_options, RV.other_values, param_bounds)
    SV_new=single_electron(None, SV.dim_dict, SV.simulation_options, SV.other_values, param_bounds)
    DCV_new=single_electron(None, DCV.dim_dict, DCV.simulation_options, DCV.other_values, param_bounds)
    if dispersion==True:
        #print(dispersion_optim_list)
        RV_new.simulation_options["dispersion_bins"]=[disp_bins]
        SV_new.simulation_options["dispersion_bins"]=[disp_bins]
        DCV_new.simulation_options["dispersion_bins"]=[disp_bins]
        in_table=dict(zip(dispersion_optim_list, [False]*len(dispersion_optim_list)))
        RV_new.def_optim_list(dispersion_optim_list)
        SV_new.def_optim_list(dispersion_optim_list)
        DCV_new.def_optim_list(dispersion_optim_list)
        params=[RV.dim_dict[param] for param in dispersion_optim_list]
    ramped_layout=go.Layout(height=plot_height)
    harmonic_layout=go.Layout(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0})
    r_tab_labels=["r_time", "r_volt", "r_volt_time", "r_fft"]
    s_tab_labels=["s_time", "s_volt", "s_volt_time", "s_r_fft"]
    d_tab_labels=["d_time", "d_volt", "d_volt_time"]
    if n_clicks>0:


        deletion_idx=[]
        if dispersion==False:
            table_data=table_init
        for q in range(0, len(table_data)):
            table_data[q]["Value"]=RV_new.dim_dict[table_data[q]["Parameter"]]
            if dispersion==True:
                if table_data[q]["Parameter"] in dispersion_optim_list:
                    in_table[table_data[q]["Parameter"]]=True
                if table_data[q]["Parameter"] in dispersed_params:
                    deletion_idx.append(q)

        if dispersion==True:
            for q in range(0, len(dispersion_optim_list)):
                if in_table[dispersion_optim_list[q]]==False:
                    table_data.append({"Parameter":dispersion_optim_list[q], "Value":RV_new.dim_dict[dispersion_optim_list[q]]})
            for q in range(0, len(deletion_idx)):
                del table_data[deletion_idx[q]]

        if "r_freeze" in freeze_buttons:
            if "r_scipy" in adaptive_buttons:
                RV_new.update_params(params)
                V=RV.e_nondim(RV_new.define_voltages())
                w0=[0, 0, V[0]]
                wsol = odeint(RV_new.current_ode_sys, w0, RV.time_vec, rtol=1e-6, atol=1e-6)
                timeseries=RV_new.i_nondim(wsol[:,0][RV_new.time_idx])
                RV_plot_voltages=V[RV_new.time_idx]
                RV_plot_times=RV_new.t_nondim(RV.time_vec[RV_new.time_idx])
            else:
                timeseries=RV_new.i_nondim(RV_new.test_vals(params, "timeseries"))
                RV_plot_times=RV_new.t_nondim(RV_new.time_vec[RV_new.time_idx])
                RV_plot_voltages=RV_new.e_nondim(RV_new.define_voltages()[RV_new.time_idx])
            r_harms=harmonics(list(range(start_harm, start_harm+num_harms)), RV_new.dim_dict["omega"], 0.05)
            ramped_harmonics=r_harms.generate_harmonics(RV_plot_times, timeseries, hanning=True)
            r_one_tail_len=(len(r_harms.exposed_f)//2)+1
            r_freqs=r_harms.exposed_f[:r_one_tail_len]
            r_fft=abs(r_harms.exposed_Y[:r_one_tail_len])

            if "no_decimation" in plot_buttons:
                r_x_args=[RV_plot_times, RV_plot_voltages, RV_plot_times[0::10], r_freqs]
                r_y_args=[timeseries, timeseries, RV_plot_voltages[0::10], r_fft]
            elif "ramped_rdp" in plot_buttons:
                simped_line=np.array(rdp_lines.rdp_controller(RV_plot_times, timeseries, max(timeseries)/5))
                sorted_idx=np.argsort(simped_line[:,0])
                simped_times=simped_line[:,0][sorted_idx]
                simped_timeseries=simped_line[:,1][sorted_idx]
                r_x_args=[simped_times, RV_plot_voltages[0::10], RV_plot_times[0::10], decimate(r_freqs, 10)]
                r_y_args=[simped_timeseries, timeseries[0::10], RV_plot_voltages[0::10], decimate(r_fft, 10)]
            elif "decimation" in plot_buttons:
                r_x_args=[x[0::10] for x in [RV_plot_times, RV_plot_voltages, RV_plot_times, r_freqs]]
                r_y_args=[x[0::10] for x in [timeseries, timeseries, RV_plot_voltages, r_fft]]

        if "s_freeze" in freeze_buttons:
            if "s_scipy" in adaptive_buttons:
                SV_new.update_params(params)
                V=SV.e_nondim(SV_new.define_voltages())
                C=SV_new.test_vals(params, "timeseries")
                w0=[0, 0, V[0]]
                wsol = odeint(SV_new.current_ode_sys, w0, SV.time_vec, rtol=1e-6, atol=1e-6)
                SV_timeseries=SV_new.i_nondim(wsol[:,0][SV_new.time_idx])
                SV_plot_voltages=V[SV_new.time_idx]
                SV_plot_times=SV_new.t_nondim(SV.time_vec[SV_new.time_idx])
            else:
                SV_timeseries=SV_new.i_nondim(SV_new.test_vals(params, "timeseries"))
                SV_plot_voltages=SV_new.e_nondim(SV_new.define_voltages()[SV_new.time_idx])
                SV_plot_times=SV_new.t_nondim(SV_new.time_vec[SV_new.time_idx])
            s_harms=harmonics(list(range(start_harm, start_harm+num_harms)), SV_new.dim_dict["omega"], 0.05)
            SV_harmonics=s_harms.generate_harmonics(SV_plot_times, SV_timeseries, hanning=False)
            s_one_tail_len=(len(s_harms.exposed_f)//2)+1
            s_freqs=s_harms.exposed_f[:s_one_tail_len]
            s_fft=np.real(s_harms.exposed_Y[:s_one_tail_len])
            s_x_args=[SV_plot_times, SV_plot_voltages, SV_plot_times, s_freqs]
            s_y_args=[SV_timeseries, SV_timeseries, SV_plot_voltages, s_fft]
        if "d_freeze" in freeze_buttons:
            if "d_scipy" in adaptive_buttons:
                DCV_new.update_params(params)
                V=DCV.e_nondim(DCV_new.define_voltages())
                w0=[0, 0, V[0]]
                wsol = odeint(DCV_new.current_ode_sys, w0, DCV.time_vec, rtol=1e-6, atol=1e-6)
                DCV_timeseries=DCV_new.i_nondim(wsol[:,0][DCV_new.time_idx])
                DCV_plot_voltages=V[DCV_new.time_idx]
                DCV_plot_times=DCV_new.t_nondim(DCV.time_vec[DCV_new.time_idx])
            else:
                DCV_timeseries=DCV_new.i_nondim(DCV_new.test_vals(params, "timeseries"))
                DCV_plot_voltages=DCV_new.e_nondim(DCV_new.define_voltages()[DCV_new.time_idx])
                DCV_plot_times=DCV_new.t_nondim(DCV_new.time_vec[DCV_new.time_idx])
            d_x_args=[running_reduction(x, 10) for x in [DCV_plot_times, DCV_plot_voltages, DCV_plot_times]]
            d_y_args=[running_reduction(x, 10) for x in [DCV_timeseries, DCV_timeseries, DCV_plot_voltages]]
        x_labels=["Time(s)", "Voltage(V)", "Time(s)", "Frequency(Hz)"]
        y_labels=["Current(A)", "Current(A)", "Voltage(V)", "Magnitude"]
        r_right_plots={}
        s_right_plots={}
        d_right_plots={}

        for i in range(0, len(r_tab_labels)):
            if "r_freeze" not in freeze_buttons:
                r_x_plot=[]
                r_y_plot=[]
            else:
                r_x_plot=r_x_args[i]
                r_y_plot=r_y_args[i]
            r_right_plots[r_tab_labels[i]]={"data": [
                {"x":r_x_plot, "y":r_y_plot , "type": "scattergl", "name":"Current sim", "render_mode":"webgl"},
            ],
            "layout": {"height":plot_height, "xaxis":{"title":{"text":x_labels[i]}}, "yaxis":{"title":{"text":y_labels[i]}}},
            }

            if "s_freeze" not in freeze_buttons:
                s_x_plot=[]
                s_y_plot=[]
            else:
                s_x_plot=s_x_args[i]
                s_y_plot=s_y_args[i]
            s_right_plots[s_tab_labels[i]]={"data": [
                {"x":s_x_plot, "y": s_y_plot, "type": "scattergl", "name": "Current sim", "render_mode":"webgl"},
            ],
            "layout": {"height":plot_height, "xaxis":{"title":{"text":x_labels[i]}}, "yaxis":{"title":{"text":y_labels[i]}}},
            }
            if i<len(d_tab_labels):
                if "d_freeze" not in freeze_buttons:
                    d_x_plot=[]
                    d_y_plot=[]
                else:
                    d_x_plot=d_x_args[i]
                    d_y_plot=d_y_args[i]
                d_right_plots[d_tab_labels[i]]={"data": [
                    {"x":d_x_plot, "y": d_y_plot, "type": "scattergl", "name": "Current sim", "render_mode":"webgl"},
                ],
                "layout": {"height":plot_height, "xaxis":{"title":{"text":x_labels[i]}}, "yaxis":{"title":{"text":y_labels[i]}}},
                }

        harmonics_dict={"ramped":{}, "sinusoidal":{}}
        for i in range(0, num_harms):
            xlabel=""
            ylabel=""
            b=0
            if i==(num_harms-1):
                xlabel="Time(s)"
                b=30
            if i==num_harms//2:
                ylabel="Current(A)"
            if "r_freeze" not in freeze_buttons:
                r_x_plot=[]
                r_y_plot=[]
            else:
                r_x_plot=RV_plot_times
                r_y_plot=np.abs(ramped_harmonics[i,:][0::10])
            harmonics_dict["ramped"]["ramped_harm_"+str(i)]={"data": [
                {"x":r_x_plot, "y":r_y_plot , "type": "line", "name": None, "render_mode":"webgl"},
            ],
            "layout": {"height":plot_height//num_harms, "margin":{"pad":0, "b":b, "t":5},"xaxis":{"title":{"text":xlabel}}, "yaxis":{"title":{"text":ylabel}} },
            }
        for i in range(0, num_harms):
            xlabel=""
            ylabel=""
            b=0
            if i==(num_harms-1):
                xlabel="Voltage(V)"
                b=40
            if i==num_harms//2:
                ylabel="Current(A)"
            if "s_freeze" not in freeze_buttons:
                s_x_plot=[]
                s_y_plot=[]
            else:
                s_x_plot=SV_plot_voltages
                s_y_plot=np.real(SV_harmonics[i,:])
            harmonics_dict["sinusoidal"]["sinusoidal_harm_"+str(i)]=({"data": [
                {"x":s_x_plot, "y": s_y_plot, "type": "line", "name": None, "render_mode":"webgl"},
            ],
            "layout": {"height":plot_height//num_harms, "margin":{"pad":0, "b":b, "t":5},"xaxis":{"title":{"text":xlabel}}, "yaxis":{"title":{"text":ylabel}} },
            })
        #print("simulation_time", time.time()-start)
        return [r_right_plots]+[s_right_plots]+[d_right_plots]+[harmonics_dict]+[table_data]

    empty_r_plots=dict(zip(r_tab_labels, [{"layout": ramped_layout}]*len(r_tab_labels)))
    empty_s_plots=dict(zip(s_tab_labels, [{"layout": ramped_layout}]*len(s_tab_labels)))
    empty_d_plots=dict(zip(d_tab_labels, [{"layout": ramped_layout}]*len(d_tab_labels)))
    harmonics_dict={"ramped":{}, "sinusoidal":{}}
    for i in range(0, num_harms):
        harmonics_dict["ramped"]["ramped_harm_"+str(i)]={"layout":harmonic_layout}
        harmonics_dict["sinusoidal"]["sinusoidal_harm_"+str(i)]={"layout":harmonic_layout}

    return [empty_r_plots]+[empty_s_plots]+[empty_d_plots]+[harmonics_dict]+[table_data]
harmonic_output=[Output('ramped_harm_'+str(x), 'figure') for x in range(start_harm, start_harm+num_harms)]+[Output('sinusoidal_harm_'+str(x), 'figure') for x in range(start_harm, start_harm+num_harms)]






@app.callback(
    harmonic_output,
    [Input("harmonic_store", "data"), Input("history_store", "modified_timestamp"), Input("reset_plots", "n_clicks")],
    [State("history_store", "data")]

)


def plot_harmonics(data, time, reset_button, existing_data):
    print(reset_button)
    if reset_button==0:
        for label in ["ramped", "sinusoidal"]:
            plot_keys=existing_data[label+"_harmonics"].keys()
            print(plot_keys)
            for key in plot_keys:
                for i in range(0, num_harms):
                    data[label][label+"_harm_"+str(i)]["data"].append(existing_data[label+"_harmonics"][key][label+"_harm_"+str(i)]["data"][0])
                    #print(existing_data[label+"_harmonics"][key][label+"_harm_"+str(i)]["data"][0].keys())
    harmonic_plots=[data["ramped"]["ramped_harm_"+str(i)] for i in range(0, num_harms)]+[data["sinusoidal"]["sinusoidal_harm_"+str(i)] for i in range(0, num_harms)]
    return harmonic_plots
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)

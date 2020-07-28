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
    'phase' : [0, 2*math.pi],
    "cap_phase":[0, 2*math.pi],
    "E0_mean":[-2,4],
    "E0_std": [1e-4,  1],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],

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
table_data={key:table_dict[key] for key in table_dict.keys()}
table_names=list(table_dict.keys())
SV_simulation_options=copy.deepcopy(simulation_options)
DCV_simulation_options=copy.deepcopy(simulation_options)
DCV_simulation_options["method"]="dcv"
SV_param_list=copy.deepcopy(RV_param_list)
changed_SV_params=["d_E", "phase", "cap_phase", "num_peaks", "original_omega", "sampling_freq"]
changed_sv_vals=[300e-3, 3*math.pi/2,  3*math.pi/2, 25, RV_param_list["omega"], 1/1000.0]
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
ramped_layout=go.Layout(height=plot_height)
harmonic_layout=go.Layout(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0})
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server
app.title = "FTACVSIM"
parameter_names=RV_param_list.keys()
parameter_sliders=[]
forbidden_params=["v", "d_E"]
class_dict=dict(zip(["ramped", "sinusoidal", "dcv"], [RV, SV, DCV]))

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



                                dbc.Button("Apply", id="submit-button-state",
                                           color="primary", block=True, n_clicks=0),
                                dbc.Button("Save plots", id="save_plot_state",
                                          color="primary", block=True, n_clicks=0),
                                dbc.Button("Reset parameters", id="reset_params",
                                       color="danger", block=True, n_clicks=0),
                                dbc.Button("Clear saved plots", id="reset_plots",
                                           color="danger", block=True, n_clicks=0)


                               ],body=True)


ramped_harm_init=[dcc.Graph(id="ramped_harm_"+str(x),figure=go.Figure(layout=dict(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0}))) for x in range(0,num_harms)]
sinusoidal_harm_init=[dcc.Graph(id="sinusoidal_harm_"+str(x),figure=go.Figure(layout=dict(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0}))) for x in range(0, num_harms)]
history_data={"ramped_plots":{}, "sinusoidal_plots":{}, "dcv_plots":{},"ramped_harmonics":{}, "sinusoidal_harmonics":{},
                "counters":{"ramped":0, "dcv":0, "sinusoidal":0}}

storage_array=[]
store_list=dict(zip(["ramped", "sinusoidal", "dcv"], [[], [], []]))
history_list=dict(zip(["ramped", "sinusoidal", "dcv"], [[], [], []]))
max_plots=2
history_dict={str(key):{} for key in range(0, max_plots)}
for e_type in ["ramped", "sinusoidal", "dcv"]:
    for exp in ["current_time", "current_voltage", "voltage_time", "fft", "harms"]:
        for history in ["", "_history"]:
            if e_type!="dcv":
                if exp=="harms":
                    for i in range(0, num_harms):
                        id_str=("_").join([e_type, exp, str(i), "store"])+history
                        if history=="":
                            store_list[e_type].append(id_str)
                            storage_array.append(dcc.Loading(dcc.Store(id=id_str), ))
                        else:
                            history_list[e_type].append(id_str)
                            storage_array.append(dcc.Loading(dcc.Store(id=id_str), ))
                else:
                    id_str=("_").join([e_type, exp,"store"])+history
                    if history=="":
                        store_list[e_type].append(id_str)
                        storage_array.append(dcc.Loading(dcc.Store(id=id_str), ))
                    else:
                        history_list[e_type].append(id_str)
                        storage_array.append(dcc.Loading(dcc.Store(id=id_str), ))

            elif e_type=="dcv":
                if exp not in ["fft", "harms"]:
                    id_str=("_").join([e_type, exp,"store"])+history
                    if history=="":
                        store_list[e_type].append(id_str)
                        storage_array.append(dcc.Loading(dcc.Store(id=id_str), ))
                    else:
                        history_list[e_type].append(id_str)
                        storage_array.append(dcc.Loading(dcc.Store(id=id_str), ))
table_init={key:table_dict[key] for key in table_dict.keys() if key not in disped_params}
table_init["Simulation"]="Current sim"
for disp in disped_params:
    table_init[disp]="*"


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
                            * If timing out is becoming a serious issue, please refer to the [github](https://github.com/HOLL95/FTACV_webapp), and attempt to run the program locally.

                            '''
                                     )
                    ],
                    fluid=True,
                )
            ],
            fluid=True,
            className="jumbotron bg-white text-dark"
        ),dcc.Store(id="table_store"),
        dbc.Col(storage_array),
        (dbc.Col([
            dbc.FormGroup(
                [
                    dbc.ListGroup([dbc.ListGroupItemHeading('Parameter values'),
                    dash_table.DataTable(
                        id='param_table',
                        css=[{'selector': '.row', 'rule': 'margin: 0'}],
                        columns=[
                            {"name": "Simulation", "id": "Simulation"},
                            *[{"name": key, "id": key} for key in table_dict.keys()]
                        ],
                        data=[table_init],
                        style_cell_conditional=[

                            *[{'if': {'column_id': 'key'},'width': '10px'} for key in table_dict.keys()],
                        ],
                        style_cell={'textAlign': 'left',
                                    'fontSize': 14, 'font-family': 'Helvetica'},
                        style_header={
                            'backgroundColor': 'white',
                            'fontWeight': 'bold',
                            'fontSize': 14,
                        },
                        style_table={
                                'overflowX': 'scroll',
                        },

                    ),
                ]
            ),])
        ])),
        dbc.Row([
        dbc.Col(dbc.ListGroup([dbc.ListGroupItemHeading("Ramped plot options"),dcc.RadioItems(
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
                {'label': 'Ramped', 'value': 'ramped_freeze'},
                {'label': 'Sinusoidal', 'value': 'sinusoidal_freeze'},
                {'label': 'DC', 'value': 'dcv_freeze'}
            ],
            value=["ramped_freeze", "sinusoidal_freeze", "dcv_freeze"],
            id="freeze_buttons",
        )])) ,
        dbc.Col(dbc.ListGroup([dbc.ListGroupItemHeading("Adaptive simulation"),
        dcc.Checklist(
            options=[
                {'label': 'Ramped', 'value': 'ramped_scipy'},
                {'label': 'Sinusoidal', 'value': 'sinusoidal_scipy'},
                {'label': 'DC', 'value': 'dcv_scipy'}
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
                dbc.Col([
                    dbc.Tabs(
                            [
                                dbc.Tab(label="Current-time", tab_id="ramped_current_time"),
                                dbc.Tab(label="Current-voltage", tab_id="ramped_current_voltage"),
                                dbc.Tab(label="Voltage-time", tab_id="ramped_voltage_time"),
                                dbc.Tab(label="Absolute FFT", tab_id="ramped_fft"),

                            ],
                            id="ramped_tabs",
                            active_tab="ramped_current_time",
                        ),
                        html.Div(id="ramped-tab-content", className="p-4"),
                        dbc.Tabs(
                                [
                                    dbc.Tab(label="Current-time", tab_id="sinusoidal_current_time"),
                                    dbc.Tab(label="Current-voltage", tab_id="sinusoidal_current_voltage"),
                                    dbc.Tab(label="Voltage-time", tab_id="sinusoidal_voltage_time"),
                                    dbc.Tab(label="Real FFT", tab_id="sinusoidal_fft"),
                                ],
                                id="sinusoidal_tabs",
                                active_tab="sinusoidal_current_voltage",
                            ),
                        html.Div(id="sinusoidal-tab-content", className="p-4"),

                        dbc.Tabs(
                                [
                                    dbc.Tab(label="Current-time", tab_id="dcv_current_time"),
                                    dbc.Tab(label="Current-voltage", tab_id="dcv_current_voltage"),
                                    dbc.Tab(label="Voltage-time", tab_id="dcv_voltage_time"),
                                ],
                                id="dcv_tabs",
                                active_tab="dcv_current_time",
                            ),
                        html.Div(id="dcv-tab-content", className="p-4"),],md=5),
                dbc.Col(
                        [html.Div(ramped_harm_init),
                        html.Div(sinusoidal_harm_init, style={"margin-top":"75px"})]
                        ,md=5),],align="top"
                ),
    ],
    fluid=True,
)
tab_names={"ramped":['ramped_current_time', 'ramped_current_voltage', 'ramped_voltage_time', 'ramped_fft'],
            "sinusoidal":['sinusoidal_current_time', 'sinusoidal_current_voltage', 'sinusoidal_voltage_time', 'sinusoidal_fft'],
            "dcv":['dcv_current_time', 'dcv_current_voltage', 'dcv_voltage_time']
}
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
memory_output=[]
memory_state=[]
current_state=[]
for experiment in ["ramped", "sinusoidal", "dcv"]:
    memory_output+=[Output(history_list[experiment][i], "data") for i in range(0, len(history_list[experiment]))]
    current_state+=[State(store_list[experiment][i], "data") for i in range(0, len(store_list[experiment]))]
    memory_state+=[State(history_list[experiment][i], "data") for i in range(0, len(store_list[experiment]))]
num_states=len(storage_array)//2
@app.callback(
    memory_output+[Output("table_store", "data"), Output("reset_plots", "n_clicks")],
    [Input("save_plot_state", "n_clicks")],
    [State("submit-button-state", "n_clicks"),State("reset_plots", "n_clicks")]+[State("freeze_buttons", "value")]+[State("table_store", "data")]+current_state+memory_state
)
def move_data_to_hidden(save_click,apply_click,clear_click, freeze_buttons,table_store, *stores):
    if clear_click>0:
        return [*[{}]*num_states,table_store ,0]
    if save_click>0 and apply_click>0:
        plot_number=str(save_click%max_plots)
        current_store=stores[:num_states]
        history_store=stores[num_states:]
        r_len=len(store_list["ramped"])
        s_r_len=r_len+len(store_list["sinusoidal"])
        current_table_stores=dict(zip(["ramped", "sinusoidal", "dcv"], [stores[0], stores[r_len], stores[s_r_len]]))
        table_experiments=[]
        for key in ["ramped", "sinusoidal", "dcv"]:
            if "parameters" in current_table_stores[key]:
                param_dict=current_table_stores[key]["parameters"]
                table_experiments.append(key)
        if len(table_experiments)==3:
            table_label="Saved sim "
        else:
            table_label_dict=dict(zip(["ramped", "sinusoidal", "dcv"], ["RV", "SV", "DCV"]))
            table_label=("+").join([table_label_dict[experiment] for experiment in table_experiments])+" sim "
        table_label+=plot_number
        if table_store is None:
            table_store={}
        table_store[table_label]=param_dict
        for i in range(0, num_states):
            if "figure_object" in current_store[i]:
                if "data" in current_store[i]["figure_object"]:
                    if plot_number not in history_store[i]:
                        history_store[i][plot_number]={"figure_object":{}}
                    history_store[i][plot_number]["figure_object"]=current_store[i]["figure_object"]
            elif "data" in current_store[i]:
                if plot_number not in history_store[i]:
                    history_store[i][plot_number]={"data":[]}
                history_store[i][plot_number]["data"]=current_store[i]["data"]
        return [*history_store,table_store, 0]
    return [*[{}]*num_states,table_store, 0]
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
    Output("param_table", "data"),
    [Input('submit-button-state', 'n_clicks'), Input("save_plot_state", "n_clicks"), Input("table_store", "data"), Input("reset_plots", "n_clicks")],
    [State("parameter_list", "value"),State("param_table", "data")]+[State(x+"_slider", "value") for x in table_names]
)
def update_current_table(n_clicks, save_click, table_store, reset, drop_down_opts, current_table,*slider_params):
    if n_clicks>0 and drop_down_opts is not None:
        if reset>0:
            return [table_init]
        dispersion=False
        dispersion_optim_list=[]
        dispersion_groups={"E_0":["E0_mean", "E0_std", "E0_skew"], "k_0":["k0_shape", "k0_scale"], "alpha":["alpha_mean", "alpha_std"]}
        dispersion_associations={"E0_mean":"E_0", "E0_std":"E_0", "E0_skew":"E_0", "k0_shape":"k_0", "k0_scale":"k_0", "alpha_mean":"alpha", "alpha_std":"alpha"}
        dispersed_params=list(set([dispersion_associations[key] for key in drop_down_opts if key in dispersion_associations.keys()]))
        if len(dispersed_params)!=0:
            dispersion=True
            for key in dispersed_params:
                dispersion_optim_list+=dispersion_groups[key]
        for i in range(0, len(slider_params)):
            if slider_params[i] is not None:
                val=slider_params[i]
                val=min(val, param_bounds[table_names[i]][1])
                val=max(val, param_bounds[table_names[i]][0])
                RV.dim_dict[RV.optim_list[i]]=val

        table_data=current_table
        if type(table_data) is list:
            table_data=table_data[0]

        table_keys=table_data.keys()
        for key in table_keys:
            if key in RV.dim_dict:
                table_data[key]=RV.dim_dict[key]
        for disp in disped_params:
            if dispersion==False:
                table_data[disp]="*"
            elif dispersion==True and disp not in dispersion_optim_list:
                table_data[disp]="*"
        if dispersion==True:
            for param in dispersed_params:
                table_data[param]="*"
        table_data=[table_data]
        if table_store is not None:
            for key in table_store.keys():
                print("table store", table_store[key])
                row={param:table_store[key]["param_list"][param] for param in table_store[key]["param_list"].keys()}
                row["Simulation"]=key
                for disp in disped_params:
                    if dispersion==False:
                        row[disp]="*"
                    elif dispersion==True and disp not in table_store[key]["disped_params"]:
                        row[disp]="*"
                if dispersion==True:
                    dispersed_params=list(set([dispersion_associations[key] for key in table_store[key]["disped_params"] if key in dispersion_associations.keys()]))
                    for param in dispersed_params:
                        row[param]="*"
                table_data.append(row)
        return table_data
    else:
        return current_table

left_inputs={experiment:[Input(store_list[experiment][i], "data") for i in range(0, len(store_list[experiment])) if "harms" not in store_list[experiment][i]] for experiment in ["ramped", "sinusoidal", "dcv"]}
left_history_states={experiment:[State(history_list[experiment][i], "data") for i in range(0, len(history_list[experiment])) if "harms" not in history_list[experiment][i]] for experiment in ["ramped", "sinusoidal", "dcv"]}

def tab_renderer(active_tab, reset_button, *args):
    if active_tab is not None and args[0] is not None:
        exp_id=args[-1]
        for exp in ["ramped", "sinusoidal", "dcv"]:
            if exp in exp_id:
                experiment_type=exp
                break
        exp_dict=dict(zip(["ramped", "sinusoidal", "dcv"], ["RV", "SV", "DCV"]))
        tab_entries=tab_names[experiment_type]
        stores=args[:(len(tab_names[experiment_type]))]
        histories=args[len(tab_names[experiment_type]):-1]
        tab_dict=dict(zip(tab_entries, stores))
        history_tab_dict=dict(zip(tab_entries, histories))
        history_keys=history_tab_dict[active_tab].keys()
        if len(history_keys)>0 and reset_button==0:
            if "data" not in tab_dict[active_tab]["figure_object"]:
                tab_dict[active_tab]["figure_object"]["data"]=[]
            for key in history_keys:
                history_tab_dict[active_tab][key]["figure_object"]["data"][0]["name"]=exp_dict[experiment_type]+" Sim" + str(key)
                tab_dict[active_tab]["figure_object"]["data"].append(history_tab_dict[active_tab][key]["figure_object"]["data"][0])
            return dcc.Graph(figure=tab_dict[active_tab]["figure_object"])
        else:
            return dcc.Graph(figure=tab_dict[active_tab]["figure_object"])
    return "Please click on a tab"


for experiment in ["ramped", "sinusoidal", "dcv"]:
    app.callback(
        Output(experiment+"-tab-content", "children"),
        [Input(experiment+"_tabs", "active_tab")]+[Input("reset_plots", "n_clicks")]+left_inputs[experiment],
        left_history_states[experiment]+
        [State(experiment+"_tabs", "id")])(tab_renderer)



@app.callback(
    [Output(x+"_slider", "value") for x in table_names],
    [Input('reset_params', 'n_clicks')]
)
def reset_parameters(button_click):

    slider_vals=[orig_table_dict[x] for x in orig_table_dict.keys()]
    return slider_vals

xlabels=["Time(s)", "Voltage(V)", "Time(s)", "Frequency(Hz)"]
ylabels=["Current(A)", "Current(A)", "Voltage(V)", "Magnitude"]
labels={"ramped":{"x":xlabels, "y":ylabels}, "sinusoidal":{"x":xlabels, "y":ylabels}, "dcv":{"x":xlabels[:-1], "y":ylabels[:-1]}}
def apply_slider_changes(n_clicks, drop_down_opts, disp_bins, freeze_buttons, adaptive_buttons, plot_buttons, exp_id, *slider_params):
    for exp in ["ramped", "sinusoidal", "dcv"]:
        if exp in exp_id:
            experiment_type=exp
            exp_class=class_dict[exp]
    if (n_clicks>0) and (experiment_type+"_freeze" in freeze_buttons):
        dispersion_optim_list=[]
        if drop_down_opts is not None:
            dispersion_groups={"E_0":["E0_mean", "E0_std", "E0_skew"], "k_0":["k0_shape", "k0_scale"], "alpha":["alpha_mean", "alpha_std"]}
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
        for i in range(0, len(slider_params)):
            if slider_params[i] is not None:
                val=slider_params[i]
                val=min(val, param_bounds[table_names[i]][1])
                val=max(val, param_bounds[table_names[i]][0])
                exp_class.dim_dict[RV.optim_list[i]]=val
        exp_class.dim_dict["original_gamma"]=exp_class.dim_dict["gamma"]
        exp_class.simulation_options["optim_list"]=[]
        if experiment_type=="sinusoidal":
            exp_class.dim_dict["original_omega"]=exp_class.dim_dict["omega"]
            exp_class.simulation_options["no_transient"]=2/exp_class.dim_dict["omega"]
            exp_class.dim_dict["d_E"]=(exp_class.dim_dict["E_reverse"]-exp_class.dim_dict["E_start"])/2
        elif experiment_type=="ramped":
            exp_class.simulation_options["no_transient"]=2/exp_class.dim_dict["omega"]
            exp_class.dim_dict["d_E"]=(exp_class.dim_dict["E_reverse"]-RV.dim_dict["E_start"])/4
        elif experiment_type=="dcv":
            exp_class.simulation_options["no_transient"]=False
        new_class=single_electron(None, exp_class.dim_dict, exp_class.simulation_options, exp_class.other_values, param_bounds)
        if dispersion==True:
            new_class.simulation_options["dispersion_bins"]=[disp_bins]
            new_class.def_optim_list(dispersion_optim_list)
            params=[new_class.dim_dict[param] for param in dispersion_optim_list]
        if experiment_type+"_scipy" in adaptive_buttons and dispersion==False:
            new_class.update_params(params)
            V=new_class.e_nondim(new_class.define_voltages())
            w0=[0, 0, V[0]]
            wsol = odeint(new_class.current_ode_sys, w0, new_class.time_vec, rtol=1e-6, atol=1e-6)
            timeseries=new_class.i_nondim(wsol[:,0][new_class.time_idx])
            plot_voltages=V[new_class.time_idx]
            plot_times=new_class.t_nondim(new_class.time_vec[new_class.time_idx])
        else:
            timeseries=new_class.i_nondim(new_class.test_vals(params, "timeseries"))
            plot_times=new_class.t_nondim(new_class.time_vec[new_class.time_idx])
            plot_voltages=new_class.e_nondim(new_class.define_voltages()[new_class.time_idx])
        if experiment_type!="dcv":
            harms=harmonics(list(range(start_harm, start_harm+num_harms)), new_class.dim_dict["omega"], 0.05)
            if experiment_type=="ramped":
                experiment_harmonics=harms.generate_harmonics(plot_times, timeseries, hanning=True)
            elif experiment_type=="sinusoidal":
                experiment_harmonics=harms.generate_harmonics(plot_times, timeseries, hanning=False)
            one_tail_len=(len(harms.exposed_f)//2)+1
            freqs=harms.exposed_f[:one_tail_len]

            if experiment_type=="ramped":
                fft=abs(harms.exposed_Y[:one_tail_len])
                if "no_decimation" in plot_buttons:
                    plot_list=[[plot_times, timeseries], [plot_voltages, timeseries], [plot_times[0::10], plot_voltages[0::10]], [freqs, fft]]
                elif "ramped_rdp" in plot_buttons:
                    simped_line=np.array(rdp_lines.rdp_controller(plot_times, timeseries, max(timeseries)/5))
                    sorted_idx=np.argsort(simped_line[:,0])
                    simped_times=simped_line[:,0][sorted_idx]
                    simped_timeseries=simped_line[:,1][sorted_idx]
                    plot_list=[[simped_times, simped_timeseries], [plot_voltages[0::10], timeseries[0::10]], [plot_times[0::10], plot_voltages[0::10]], [freqs, fft]]
                elif "decimation" in plot_buttons:
                    plot_list=[[plot_times[0::10], timeseries[0::10]], [plot_voltages[0::10], timeseries[0::10]], [plot_times[0::10], plot_voltages[0::10]], [freqs[0::10], fft[0::10]]]
            elif experiment_type=="sinusoidal":
                fft=np.real(harms.exposed_Y[:one_tail_len])
                plot_list=[[plot_times, timeseries], [plot_voltages, timeseries], [plot_times, plot_voltages], [freqs, fft]]
        else:
            plot_list=[[plot_times[0::10], timeseries[0::10]], [plot_voltages[0::10], timeseries[0::10]], [plot_times[0::10], plot_voltages[0::10]]]

        non_harm_plots=[{"figure_object":{"data": [
                        {"x":plot_list[i][0], "y":plot_list[i][1] , "type": "scattergl", "name":"Current sim", "render_mode":"webgl"}
                        ],
                        "layout":
                        {"height":plot_height, "xaxis":{"title":{"text":labels[experiment_type]["x"][i]}}, "yaxis":{"title":{"text":labels[experiment_type]["y"][i]}}}
                        ,},"parameters":{"param_list":dict(zip(RV.optim_list, slider_params)),"disped_params":dispersion_optim_list, "experiment":experiment_type}} for i in range(0, len(plot_list))]
        return_arg=non_harm_plots
        if experiment_type!="dcv":
            for i in range(0, num_harms):
                xlabel=""
                ylabel=""
                b=0
                if i==(num_harms-1):
                    xlabel="Time(s)"
                    b=30
                if i==num_harms//2:
                    ylabel="Current(A)"


                if experiment_type=="ramped":
                    x_plot=plot_times
                    y_plot=np.abs(experiment_harmonics[i,:][0::10])
                else:
                    x_plot=plot_voltages
                    y_plot=np.real(experiment_harmonics[i,:])
                return_arg.append({"data": [
                    {"x":x_plot, "y":y_plot , "type": "line", "name": "Current sim", "render_mode":"webgl"},
                ],
                "layout": {"height":plot_height//num_harms, "margin":{"pad":0, "b":b, "t":5},"xaxis":{"title":{"text":xlabel}}, "yaxis":{"title":{"text":ylabel}} },
                })
        return return_arg
    else:
        if experiment_type=="dcv":
            return [{"figure_object":{"layout": ramped_layout}}]*len(labels["dcv"]["y"])
        else:
            return [{"figure_object":{"layout": ramped_layout}}]*len(labels["ramped"]["y"])+[{"layout": harmonic_layout}]*(num_harms)
for experiment in ["ramped", "sinusoidal", "dcv"]:
    output=[Output(store_list[experiment][i], "data") for i in range(0, len(store_list[experiment]))]
    input=[Input('submit-button-state', 'n_clicks')]
    state=[State(component_id="parameter_list", component_property="value")]+\
    [State("dispersion_slider", "value")]+\
    [State("freeze_buttons", "value")]+\
    [State("adaptive_buttons", "value")]+\
    [State("plot_buttons", "value")]+\
    [State(store_list[experiment][0], "id")]+\
    [State(x+"_slider", "value") for x in table_names]
    app.callback(output,input,state)(apply_slider_changes)



def plot_harmonics(current_harm_store, reset_button, history_harm_store, exp_id):
    if reset_button==0:
        for exp in ["ramped", "sinusoidal", "dcv"]:
            if exp in exp_id:
                experiment_type=exp
                break
        exp_dict=dict(zip(["ramped", "sinusoidal", "dcv"], ["RV", "SV", "DCV"]))
        num_plots=history_harm_store.keys()
        if "data" not in current_harm_store:
            current_harm_store["data"]=[]
        for key in num_plots:
            history_harm_store[key]["data"][0]["name"]=exp_dict[experiment_type]+" Sim" + str(key)
            current_harm_store["data"].append(history_harm_store[key]["data"][0])
    if current_harm_store is None:
        current_harm_store={"data":[],"layout":harmonic_layout}
    return current_harm_store
for experiment in ["ramped", "sinusoidal"]:
    for i in range(0, num_harms):
        app.callback(
            Output(experiment+"_harm_"+str(i), "figure"),
            [Input(experiment+"_harms_"+str(i)+"_store", "data"),Input("reset_plots", "n_clicks")],
            [State(experiment+"_harms_"+str(i)+"_store_history", "data"), State(experiment+"_harms_"+str(i)+"_store_history", "id")]
        )(plot_harmonics)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)

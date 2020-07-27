import dash_table
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from flask_caching.backends import FileSystemCache
from dash_extensions.callback import CallbackCache, Trigger
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
cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))
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
                            storage_array.append(dcc.Store(id=id_str))
                        else:
                            history_list[e_type].append(id_str)
                            storage_array.append(dcc.Store(id=id_str, data=history_dict))
                else:
                    id_str=("_").join([e_type, exp,"store"])+history
                    if history=="":
                        store_list[e_type].append(id_str)
                        storage_array.append(dcc.Store(id=id_str))
                    else:
                        history_list[e_type].append(id_str)
                        storage_array.append(dcc.Store(id=id_str, data=history_dict))

            elif e_type=="dcv":
                if exp not in ["fft", "harms"]:
                    id_str=("_").join([e_type, exp,"store"])+history
                    if history=="":
                        store_list[e_type].append(id_str)
                        storage_array.append(dcc.Store(id=id_str))
                    else:
                        history_list[e_type].append(id_str)
                        storage_array.append(dcc.Store(id=id_str, data=history_dict))
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
        dbc.Row(storage_array),
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
        cc.callback(
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
@cc.callback(
    memory_output+[Output("reset_plots", "n_clicks")],
    [Input("save_plot_state", "n_clicks")],
    [State("submit-button-state", "n_clicks"),State("reset_plots", "n_clicks")]+[State("freeze_buttons", "value")]+[State("table_store", "data")]+current_state+memory_state
)
def move_data_to_hidden(save_click,apply_click,clear_click, freeze_buttons,table_store, *stores):
    if clear_click>0:
        return [*[{}]*num_states,0]
    if save_click>0 and apply_click>0:
        plot_number=str(save_click%max_plots)
        current_store=stores[:num_states]
        history_store=stores[num_states:]
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
@cc.callback(
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


left_inputs={experiment:[Input(store_list[experiment][i], "data") for i in range(0, len(store_list[experiment])) if "harms" not in store_list[experiment][i]] for experiment in ["ramped", "sinusoidal", "dcv"]}
left_history_states={experiment:[State(history_list[experiment][i], "data") for i in range(0, len(history_list[experiment])) if "harms" not in history_list[experiment][i]] for experiment in ["ramped", "sinusoidal", "dcv"]}
def tab_renderer(active_tab, reset_button, *args):
    if active_tab is not None and args[0] is not None:
        exp_id=args[-1]
        for exp in ["ramped", "sinusoidal", "dcv"]:
            if exp in exp_id:
                experiment_type=exp
                break
            return dcc.Graph(figure=tab_dict[active_tab]["figure_object"])
        else:
            return dcc.Graph(figure=tab_dict[active_tab]["figure_object"])
    return "Please click on a tab"


for experiment in ["ramped", "sinusoidal", "dcv"]:
    cc.callback(
        Output(experiment+"-tab-content", "children"),
        [Input(experiment+"_tabs", "active_tab")]+[Input("reset_plots", "n_clicks")]+left_inputs[experiment],
        left_history_states[experiment]+
        [State(experiment+"_tabs", "id")])(tab_renderer)



@cc.callback(
    [Output(x+"_slider", "value") for x in table_names],
    [Input('reset_params', 'n_clicks')]
)
def reset_parameters(button_click):

    slider_vals=[orig_table_dict[x] for x in orig_table_dict.keys()]
    return slider_vals

xlabels=["Time(s)", "Voltage(V)", "Time(s)", "Frequency(Hz)"]
ylabels=["Current(A)", "Current(A)", "Voltage(V)", "Magnitude"]
labels={"ramped":{"x":xlabels, "y":ylabels}, "sinusoidal":{"x":xlabels, "y":ylabels}, "dcv":{"x":xlabels[:-1], "y":ylabels[:-1]}}
def apply_slider_changes(n_clicks, exp_id):
    for exp in ["ramped", "sinusoidal", "dcv"]:
        if exp in exp_id:
            experiment_type=exp
            exp_class=class_dict[exp]
    if n_clicks>0:
        # slow process goes here
        plot_1=np.arange(0, 1e5)
        plot_2=np.arange(0, 1e5)
        plot_list=[[plot_1, plot_2]]*len(labels[experiment_type])
        non_harm_plots=[{"figure_object":{"data": [
                        {"x":plot_list[i][0], "y":plot_list[i][1] , "type": "scattergl", "name":"Current sim", "render_mode":"webgl"}
                        ],
                        "layout":
                        {"height":plot_height,}
                        }} for i in range(0, len(plot_list))]
        return_arg=non_harm_plots
        if experiment_type!="dcv":
            for i in range(0, num_harms):
                return_arg.append({"data": [
                    {"x":plot_1, "y":plot_2, "type": "line", "name": "Current sim", "render_mode":"webgl"},
                ],
                "layout": {"height":plot_height//num_harms, "margin":{"pad":0, "b":b, "t":5}},
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
    state=[State(store_list[experiment][0], "id")]
    cc.cached_callback(output,input,state)(apply_slider_changes)



def plot_harmonics(current_harm_store, reset_button, history_harm_store, exp_id):
    if reset_button==0:
        for exp in ["ramped", "sinusoidal", "dcv"]:
            if exp in exp_id:
                experiment_type=exp
                break
        print(current_harm_store)
    if current_harm_store is None:
        current_harm_store={"data":[],"layout":harmonic_layout}
    return current_harm_store
for experiment in ["ramped", "sinusoidal"]:
    for i in range(0, num_harms):
        cc.callback(
            Output(experiment+"_harm_"+str(i), "figure"),
            [Input(experiment+"_harms_"+str(i)+"_store", "data"),Input("reset_plots", "n_clicks")],
            [State(experiment+"_harms_"+str(i)+"_store_history", "data"), State(experiment+"_harms_"+str(i)+"_store_history", "id")]
        )(plot_harmonics)

cc.register(app)
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)

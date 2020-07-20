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
external_stylesheets = [dbc.themes.BOOTSTRAP]

RV_param_list={
    "E_0":-0.3,
    'E_start':  -500e-3, #(starting dc voltage - V)
    'E_reverse':100e-3,
    'omega':8.88480830076,  #    (frequency Hz)
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
    'k_0': 10, #(reaction rate s-1)
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
default_params=RV_param_list.keys()
forbidden_params=["E0_mean", "E0_std", "E0_skew","k0_scale", "k0_shape",
                    "alpha_mean", "alpha_std", "E_start", "E_reverse",
                    "original_gamma", "v", "d_E", "area", "sampling_freq", "phase", "cap_phase"]
default_dict=copy.deepcopy(RV_param_list)
for param in forbidden_params:
    del default_dict[param]

param_bounds={
    "E_start":[-2, 2],
    "E_reverse":[0, 4],
    "v":[-1e-2, 1e-2],
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
table_names=list(param_bounds.keys())
dispersion_params=["E0_mean", "E0_std", "E0_skew","k0_scale", "k0_shape",
                    "alpha_mean", "alpha_std", "d_E"]
table_data=[{"Parameter":key, "Value":RV_param_list[key]} for key in table_names if key not in dispersion_params]
basic_optim_list=default_dict.keys()
SV_simulation_options=copy.deepcopy(simulation_options)
SV_param_list=copy.deepcopy(RV_param_list)
del SV_param_list["v"]
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
RV_plot_times=RV.t_nondim(RV.time_vec[RV.time_idx])
SV_plot_times=SV.t_nondim(SV.time_vec[SV.time_idx])
RV.def_optim_list(list(basic_optim_list))
SV_plot_voltages=SV.e_nondim(SV.define_voltages()[SV.time_idx])
times=SV.test_vals([], "timeseries")
SV.def_optim_list(list(basic_optim_list))


plot_height=max(75*num_harms, 450)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "FTACVSim"
parameter_names=RV_param_list.keys()
parameter_sliders=[]
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
controls = dbc.Card([dbc.FormGroup([
                                dbc.Label("Parameters"),
                                dcc.Dropdown(
                                                id='parameter_list',
                                                options=[{'label': x, 'value': x} for x in table_names],
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
                                            data=table_data,
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
                        html.H1("FTACV Sim", className="display-3"),
                        html.P(
                            "Interactively simulate Ramped and sinusoidal voltammetry experiments ",
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
                [State(component_id="parameter_list", component_property="value")]
                )

def apply_param_changes(n_clicks, *inputs):
    states=inputs[:-2]
    table_data=inputs[-2]
    drop_down_opts=inputs[-1]
    input_options=["E_start", "E_reverse", "v", "sampling_freq", "area", "omega"]
    re_initialise=False
    sim_params_dict=dict(zip(RV.optim_list, [0]*len(RV.optim_list)))

    for i in range(0, len(table_names)):
        if states[i] is not None:
            val=states[i]
            val=min(val, param_bounds[table_names[i]][1])
            val=max(val, param_bounds[table_names[i]][0])
            sim_params_dict[table_names[i]]=val
    #        if table_names[i] in input_options:
    #            if RV_param_list[table_names[i]]!=val:
    #                RV_param_list[table_names[i]]=val
    #                if table_names[i]!="v":
    #                    SV_param_list[table_names[i]]=val
    #                re_intitialise=True
    #        if table_names[i]=="omega":
    #            if (val>(10*SV.dim_dict["original_omega"]) or val<(0.1*SV.dim_dict["original_omega"])):
    #                SV_param_list["original_omega"]=val
    #                SV.simulation_options["time_start"]=2/val
    #                re_initialise=True
    #        if table_names[i]=="gamma" and (val>(10*SV.dim_dict["original_gamma"]) or val<(0.1*SV.dim_dict["original_gamma"])):
    #            SV_param_list["original_gamma"]=val
    #            RV_param_list["original_gamma"]=val
    #            re_initialise=True
        else:
            sim_params_dict[table_names[i]]=RV.dim_dict[table_names[i]]
    #if re_initialise==True:
    #    SV_param_list["d_E"]=(SV_param_list["E_reverse"]-SV_param_list["E_start"])/2
    #    RV_param_list["d_E"]=(RV_param_list["E_reverse"]-RV_param_list["E_start"])/4
    #    RV.calculate_times(RV_param_list)
    #    SV.calculate_times(SV_param_list)
    params=[sim_params_dict[key] for key in RV.optim_list]
    ramped_layout=go.Layout(height=plot_height)
    harmonic_layout=go.Layout(height=plot_height//num_harms, margin={"pad":0, "b":0, "t":0})
    if n_clicks>0:
        timeseries=RV.i_nondim(RV.test_vals(params, "timeseries"))
        for q in range(0, len(table_data)):
            table_data[q]["Value"]=RV.dim_dict[table_data[q]["Parameter"]]
        SV_timeseries=np.zeros(len(timeseries))#SV.i_nondim(SV.test_vals(params, "timeseries"))
        SV_plot_voltages=SV.e_nondim(SV.define_voltages()[SV.time_idx])
        print("simulated")
        harms=harmonics(list(range(start_harm, start_harm+num_harms)), RV.dim_dict["omega"], 0.05)
        print(len(RV_plot_times), len(timeseries))
        print(len(SV_plot_times), len(SV_timeseries))
        ramped_harmonics=harms.generate_harmonics(RV_plot_times, timeseries, hanning=True)
        SV_harmonics=harms.generate_harmonics(SV_plot_times, SV_timeseries, hanning=False)
        print("generated_ramped")



        print("harmonics_generated")



        return_arg=[{"data": [
            {"x":RV_plot_times, "y": timeseries, "type": "line", "name": "Ramped", "render_mode":"webgl"},
        ],
        "layout": ramped_layout,
        }]
        for i in range(0, num_harms):
            return_arg.append({"data": [
                {"x":RV_plot_times, "y": np.abs(ramped_harmonics[i,:])[0::32], "type": "line", "name": "Ramped_harm"+str(i), "render_mode":"webgl"},
            ],
            "layout": harmonic_layout,
            })
        return_arg.append({"data": [
            {"x":SV_plot_voltages, "y": SV_timeseries, "type": "line", "name": "SV", "render_mode":"webgl"},
        ],
        "layout": ramped_layout,
        })

        for i in range(0, num_harms):
            return_arg.append({"data": [
                {"x":SV_plot_voltages, "y": np.real(SV_harmonics[i,:]), "type": "line", "name": "sv_harm"+str(i), "render_mode":"webgl"},
            ],
            "layout": harmonic_layout,
            })
        return_arg.append(table_data)
        print("laid out")
        return return_arg


    return [{"layout": ramped_layout}]+[{"layout": harmonic_layout}]*(num_harms)+[{"layout": ramped_layout}]+[{"layout": harmonic_layout}]*(num_harms)+[table_data]



if __name__ == '__main__':
    app.run_server(debug=True)

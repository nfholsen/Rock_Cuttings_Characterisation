from dash import Dash, html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import shutil
import random

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)


app.layout = html.Div(
    [   
        html.Br(),
        html.H2('Rock cuttings prediction', style={'width':'50%','justify':'center'},),
        html.Br(),

        html.Div(
            dcc.Markdown(
                """
                Description :
                In total there will be for choice, please complete the test, ie the 100 images to classify
                Cliking on the help button marked with the **?**

                Four test are available and the results will be saved in the folder created when the file is launched.

                The four tests are :
                * Lab/Lab : Laboratory cuttings classifications given examples of laboratory cuttings slices
                * Lab/Borehole : Laboratory cuttings classifications given examples of borehole cuttings slices
                * Borehole/Borehole : Borehole cuttings classifications given examples of borehole cuttings slices
                * Borehole/Lab : Borehole cuttings classifications given examples of laboratory cuttings slices

                Once the test is done the outputs are save in the corresponding csv file. If you redo the test the slices to classify will be the same but the order different.

                Good luck
                """
            ),
        ),

        html.Br(),

        # Dropdown menu to select the test we want to conduct
        html.Div(
            [
                dcc.Dropdown(
                    [
                        {'label':'Lab/Lab','value':'data/lab/train_test_train_mar.csv~data/lab/train_test_test_mar.csv'},
                        {'label':'Lab/Borehole','value':'data/lab/train_test_train_mar.csv~data/borehole/train_test_test_mar.csv'},
                        {'label':'Borehole/Borehole','value':'data/borehole/train_test_train_mar.csv~data/borehole/train_test_test_mar.csv'},
                        {'label':'Borehole/Lab','value':'data/borehole/train_test_train_mar.csv~data/lab/train_test_test_mar.csv'},
                    ],
                    id='select_df',
                ),
            ]
        ),
        dcc.Store(data={},id='store_df'),
        dcc.Store(data=[],id='random_list'),
        html.Br(),

        html.Div(
            [
                # Image to classify
                dbc.Card(
                    [],
                    id='img_to_classify',
                ),

                html.Br(),

                # Selection of the label with the button
                html.Div(
                    dbc.Row(
                        [
                            dbc.Col(dbc.Button('Rock 1 - BL', id='but_0')), # 0
                            dbc.Col(dbc.Button('Rock 2 - GN', id='but_1')), # 1
                            dbc.Col(dbc.Button('Rock 3 - ML', id='but_2')), # 2
                            dbc.Col(dbc.Button('Rock 4 - MS', id='but_3')), # 3
                            dbc.Col(dbc.Button('Rock 5 - OL', id='but_4')), # 4
                        ]
                    ),
                ),

                html.Br(),

                # Counter
                html.Div(
                    children=[],
                    id='counter',
                ),
            ],
            style={'text-align':'center'}
        ),
        html.Br(),
        # Accordion to display examples
        html.Div(
            [ 
                dbc.Accordion(
                    [
                        dbc.AccordionItem(children=[html.P('No test selected yet')], title="Click to display examples",id='examples'),
                    ],start_collapsed=False
                )
            ]
        ),
    ]
)

# Select from the menu the experiment - return the examples and the df with the data for the test
@app.callback(
    Output('examples','children'), # For the examples
    Output('store_df','data'), # For the test
    Output('random_list','data'),
    Input('select_df','value'),
)
def store_data(select_df):
    # def move_img(df):
    #     paths = df['Paths'].to_list()
    #     clean_paths = []
    #     for path in paths:
    #         source = path
    #         destination = "assets/{}".format(path[3:].replace('/','_').replace('\\','_'))
    #         shutil.copy(source,destination)
        
    #         clean_paths.append(destination)
    #     return clean_paths

    if select_df is None:
        return [], {}
    else:
        examples = select_df.split('~')[0]
        test = select_df.split('~')[1]

        # Get .png path for the examples
        df_examples = pd.read_csv(examples,index_col=0)
        #df_examples = df_examples.groupby('Label').sample(10,replace=False,random_state=0).reset_index(drop=True)
        # clean_paths = move_img(df_examples) # Return clean_paths
        # df_examples.to_csv(examples)

        # Get .png for the test
        df_test = pd.read_csv(test,index_col=0)
        # Select 100 examples, 20 for each class randomly but fixed seed
        #df_test = df_test.groupby('Label').sample(20,replace=False,random_state=0).reset_index(drop=True)
        # clean_paths = move_img(df_test) # Return clean_paths
        # df_test['Preds'] = -1
        # df_test.to_csv(test)


    #html.Img(src=app.get_asset_url('logo.png'), style={'height':'10%', 'width':'10%'}

        layout_examples = html.Div(
            [
                html.Br(),
                html.P('BL'),
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=df_examples['Paths_Test'].iloc[j], style={'height':'100px', 'width':'100px'}),className="col-md-1") #dbc.Col(dbc.Card(dbc.CardImg(src=df_examples['Paths_Test'].iloc[j+i*10]),style={'height':'100px','width':'100px'}),className="col-md-1")
                        for j in range(10)
                    ] 
                ),
                html.Br(),
                html.P('GN'),
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=df_examples['Paths_Test'].iloc[j+10], style={'height':'100px', 'width':'100px'}),className="col-md-1") #dbc.Col(dbc.Card(dbc.CardImg(src=df_examples['Paths_Test'].iloc[j+i*10]),style={'height':'100px','width':'100px'}),className="col-md-1")
                        for j in range(10)
                    ] 
                ), 
                html.Br(),
                html.P('ML'),           
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=df_examples['Paths_Test'].iloc[j+20], style={'height':'100px', 'width':'100px'}),className="col-md-1") #dbc.Col(dbc.Card(dbc.CardImg(src=df_examples['Paths_Test'].iloc[j+i*10]),style={'height':'100px','width':'100px'}),className="col-md-1")
                        for j in range(10)
                    ] 
                ),
                html.Br(),
                html.P('MS'),
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=df_examples['Paths_Test'].iloc[j+30], style={'height':'100px', 'width':'100px'}),className="col-md-1") #dbc.Col(dbc.Card(dbc.CardImg(src=df_examples['Paths_Test'].iloc[j+i*10]),style={'height':'100px','width':'100px'}),className="col-md-1")
                        for j in range(10)
                    ] 
                ),
                html.Br(),
                html.P('OL'),
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=df_examples['Paths_Test'].iloc[j+40], style={'height':'100px', 'width':'100px'}),className="col-md-1") #dbc.Col(dbc.Card(dbc.CardImg(src=df_examples['Paths_Test'].iloc[j+i*10]),style={'height':'100px','width':'100px'}),className="col-md-1")
                        for j in range(10)
                    ] 
                )
            ]
        )
        return layout_examples, df_test[['Paths_Test','Label','Preds']].to_dict(), random.sample(range(0,df_test.shape[0]), df_test.shape[0])


@app.callback(
    Output('counter','children'),
    Output('img_to_classify','children'),
    Input('store_df','data'),
    Input('random_list','data'),
    Input('but_0','n_clicks'),
    Input('but_1','n_clicks'),
    Input('but_2','n_clicks'),
    Input('but_3','n_clicks'),
    Input('but_4','n_clicks')
)
def updates(data,random_list,n0,n1,n2,n3,n4):

    button_id = ctx.triggered_id if not None else 'No clicks yet'        
    #print(button_id)


    counts = [ctx.inputs[but] if ctx.inputs[but] is not None else 0 for but in ['but_0.n_clicks','but_1.n_clicks','but_2.n_clicks','but_3.n_clicks','but_4.n_clicks']]
    print(counts)
    counts = np.sum(counts)

    paths = pd.DataFrame.from_dict(data)['Paths_Test']

    return html.Div([f'{np.sum(counts)+1}/100']), dbc.Col(html.Img(src=paths[random_list[counts]], style={'height':'500px', 'width':'500px'}),className="col-md-6") #dbc.Col(dbc.Card(dbc.CardImg(src=df_examples['Paths_Test'].iloc[j+i*10]),style={'height':'100px','width':'100px'}),className="col-md-1")

if __name__ == '__main__':
    app.run_server(debug=True)
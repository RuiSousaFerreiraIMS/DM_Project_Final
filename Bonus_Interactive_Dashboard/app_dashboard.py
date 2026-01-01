import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.colors as pc

# ----------------------
# Upload de data
# ----------------------
df = pd.read_csv("customers_with_clusters_final.csv")

df_viz = df[['Customer Name', 'Gender', 'Province or State', 'Education', 'Income', 'Marital Status',
             'NumFlights', 'PointsAccumulated', 'CLV', 'merged_labels', 'Latitude', 'Longitude']].copy()

df_viz['Cluster'] = df_viz['merged_labels'].astype(int).astype(str)

# UX/UI / Colors
num_clusters = df_viz['Cluster'].nunique()
colors = pc.qualitative.Plotly * ((num_clusters // len(pc.qualitative.Plotly)) + 1)
cluster_colors = colors[:num_clusters]

# ----------------------
# INIT
# ----------------------
app = dash.Dash(__name__)
app.title = "Customer Cluster Dashboard"

# ----------------------
# Layout
# ----------------------
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Customer Cluster Dashboard", style={'textAlign': 'center', 'marginBottom': '5px', 'color': '#333'}),
        html.P("Interactive exploration of customer clusters", style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#555'})
    ], style={'backgroundColor': '#e0e0e0', 'padding': '15px', 'borderRadius': '10px'}),

    # Download button
    html.Div([
        html.Button("Download Filtered CSV", id="btn-download", n_clicks=0,
                    style={'marginBottom': '20px', 'backgroundColor': '#4C9F70', 'color': 'white',
                           'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
        dcc.Download(id="download-dataframe-csv")
    ], style={'textAlign': 'center'}),

    # Filters
    html.Div([
        # Cada filtro
        html.Div([html.Label("Gender", style={'width': '100px', 'display': 'inline-block'}),
                  dcc.Dropdown(
                      id='gender-filter',
                      options=[{'label': g, 'value': g} for g in df_viz['Gender'].dropna().unique()],
                      multi=True,
                      placeholder="Select",
                      style={'width': '180px'}
                  )], style={'marginRight': '20px'}),

        html.Div([html.Label("State", style={'width': '100px', 'display': 'inline-block'}),
                  dcc.Dropdown(
                      id='state-filter',
                      options=[{'label': s, 'value': s} for s in df_viz['Province or State'].dropna().unique()],
                      multi=True,
                      placeholder="Select",
                      style={'width': '180px'}
                  )], style={'marginRight': '20px'}),

        html.Div([html.Label("Education", style={'width': '100px', 'display': 'inline-block'}),
                  dcc.Dropdown(
                      id='education-filter',
                      options=[{'label': e, 'value': e} for e in df_viz['Education'].dropna().unique()],
                      multi=True,
                      placeholder="Select",
                      style={'width': '180px'}
                  )], style={'marginRight': '20px'}),

        html.Div([html.Label("Marital Status", style={'width': '120px', 'display': 'inline-block'}),
                  dcc.Dropdown(
                      id='marital-filter',
                      options=[{'label': m, 'value': m} for m in df_viz['Marital Status'].dropna().unique()],
                      multi=True,
                      placeholder="Select",
                      style={'width': '180px'}
                  )], style={'marginRight': '20px'}),

        html.Div([html.Label("Income", style={'width': '100px', 'display': 'inline-block'}),
                  dcc.RangeSlider(
                      id='income-filter',
                      min=df_viz['Income'].min(),
                      max=df_viz['Income'].max(),
                      step=1000,
                      value=[df_viz['Income'].min(), df_viz['Income'].max()],
                      marks={int(i): str(int(i)) for i in range(int(df_viz['Income'].min()),
                                                                int(df_viz['Income'].max()) + 1,
                                                                int((df_viz['Income'].max() - df_viz['Income'].min()) / 5))},
                      tooltip={"placement": "bottom", "always_visible": True},
                      allowCross=False,
                      updatemode='mouseup'
                  )], style={'width': '300px'})
    ], style={'padding': '15px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
              'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '20px'}),

    # Plots
    html.Div([
        html.Div([
            html.Div([dcc.Graph(id='scatter-3d', style={'height': '500px'})], style={'flex': '1', 'minWidth': '40%', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='map-clusters', style={'height': '500px'})], style={'flex': '1.5', 'minWidth': '55%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'width': '95%', 'margin': 'auto'}),

        html.Div([
            html.Div([dcc.Graph(id='bar-cluster', style={'height': '350px'})], style={'flex': '1', 'minWidth': '48%', 'marginRight': '4%'}),
            html.Div([dcc.Graph(id='hist-clv', style={'height': '350px'})], style={'flex': '1', 'minWidth': '48%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'width': '95%', 'margin': 'auto', 'marginTop': '20px'})
    ]),

    # Footer
    html.Footer(
        "Project by Group 23 – Data Mining Course, NOVA IMS – MSc in Data Science & Business Analytics",
        style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#e0e0e0', 'marginTop': '30px', 'borderRadius': '5px'}
    )
], style={'backgroundColor': '#f5f5f5', 'padding': '20px'})


# ----------------------
# Callback download CSV
# ----------------------
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    State('gender-filter', 'value'),
    State('state-filter', 'value'),
    State('education-filter', 'value'),
    State('marital-filter', 'value'),
    State('income-filter', 'value'),
    prevent_initial_call=True
)
def download_filtered_csv(n_clicks, selected_genders, selected_states, selected_edu, selected_marital, income_range):
    df_filtered = df_viz.copy()

    if selected_genders:
        df_filtered = df_filtered[df_filtered['Gender'].isin(selected_genders)]
    if selected_states:
        df_filtered = df_filtered[df_filtered['Province or State'].isin(selected_states)]
    if selected_edu:
        df_filtered = df_filtered[df_filtered['Education'].isin(selected_edu)]
    if selected_marital:
        df_filtered = df_filtered[df_filtered['Marital Status'].isin(selected_marital)]
    if income_range:
        df_filtered = df_filtered[(df_filtered['Income'] >= income_range[0]) &
                                  (df_filtered['Income'] <= income_range[1])]

    return dcc.send_data_frame(df_filtered.to_csv, "filtered_customers.csv", index=False)

# ----------------------
# Callback plots
# ----------------------
@app.callback(
    [Output('scatter-3d', 'figure'),
     Output('map-clusters', 'figure'),
     Output('bar-cluster', 'figure'),
     Output('hist-clv', 'figure')],
    [Input('gender-filter', 'value'),
     Input('state-filter', 'value'),
     Input('education-filter', 'value'),
     Input('marital-filter', 'value'),
     Input('income-filter', 'value')]
)
def update_graphs(selected_genders, selected_states, selected_edu, selected_marital, income_range):
    df_filtered = df_viz.copy()
    if selected_genders:
        df_filtered = df_filtered[df_filtered['Gender'].isin(selected_genders)]
    if selected_states:
        df_filtered = df_filtered[df_filtered['Province or State'].isin(selected_states)]
    if selected_edu:
        df_filtered = df_filtered[df_filtered['Education'].isin(selected_edu)]
    if selected_marital:
        df_filtered = df_filtered[df_filtered['Marital Status'].isin(selected_marital)]
    if income_range:
        df_filtered = df_filtered[(df_filtered['Income'] >= income_range[0]) &
                                  (df_filtered['Income'] <= income_range[1])]

    if df_filtered.empty:
        empty_fig = px.scatter_3d(pd.DataFrame({'NumFlights': [], 'PointsAccumulated': [], 'CLV': [], 'Cluster': []}),
                                  x='NumFlights', y='PointsAccumulated', z='CLV')
        empty_map = px.scatter_mapbox(pd.DataFrame({'Latitude': [], 'Longitude': [], 'Cluster': []}),
                                      lat='Latitude', lon='Longitude')
        empty_bar = px.bar(pd.DataFrame({'Cluster': [], 'Count': []}), x='Cluster', y='Count')
        empty_hist = px.histogram(pd.DataFrame({'CLV': [], 'Cluster': []}), x='CLV')
        return empty_fig, empty_map, empty_bar, empty_hist

    # Scatter 3D
    scatter_fig = px.scatter_3d(
        df_filtered,
        x='NumFlights', y='PointsAccumulated', z='CLV',
        color='Cluster',
        color_discrete_sequence=cluster_colors,
        hover_data=['Customer Name', 'Gender', 'Education', 'Income', 'NumFlights', 'CLV']
    )
    scatter_fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),  # margem superior suficiente para o título
        paper_bgcolor='white',
        title={
            'text': "3D Scatter of Customer Behavior",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#333'}
        }
    )

    # Mapbox
    provinces = df_filtered.groupby('Province or State').agg({
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Cluster': 'count'
    }).reset_index()

    # New color palette
    state_colors = px.colors.qualitative.Bold * ((len(provinces) // len(px.colors.qualitative.Bold)) + 1)
    color_map = {province: state_colors[i] for i, province in enumerate(provinces['Province or State'])}

    # Hover text
    hover_text = []
    for province, data in df_filtered.groupby('Province or State'):
        counts = data['Cluster'].value_counts()
        text = f"{province}<br>" + "<br>".join([f"Cluster {c}: {v}" for c, v in counts.items()])
        hover_text.append(text)

    map_fig = px.scatter_mapbox(
        provinces,
        lat='Latitude',
        lon='Longitude',
        size='Cluster',
        color='Province or State',
        color_discrete_map=color_map,
        hover_name='Province or State',
        hover_data={'Hover': hover_text},
        zoom=3,
        size_max=50
    )
    map_fig.update_layout(
        mapbox_style="open-street-map",
        title="Customer Distribution by Province/State"
    )

    # Bar chart clusters
    bar_fig = px.bar(df_filtered.groupby('Cluster').size().reset_index(name='Count'),
                     x='Cluster', y='Count',
                     color='Cluster', color_discrete_sequence=cluster_colors)
    bar_fig.update_layout(title="Number of Customers per Cluster")

    # Histogram CLV
    hist_fig = px.histogram(df_filtered, x='CLV', nbins=20,
                            color='Cluster', barmode='overlay', color_discrete_sequence=cluster_colors)
    hist_fig.update_layout(title="Customer Lifetime Value (CLV) Distribution")

    return scatter_fig, map_fig, bar_fig, hist_fig

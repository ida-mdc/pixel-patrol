# my_app/app.py
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Import page layouts and callback registration functions
from pages import processing, report

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the overall app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='project-data-store'),

    dbc.Nav(
        [
            dbc.NavLink("Processing", href="/processing", active="exact"),
            dbc.NavLink("Report", href="/report", active="exact"),
        ],
        pills=True,
        className="mt-3 mb-3"
    ),
    html.Hr(),
    html.Div(id='page-content')
])

# Callback to update page content based on URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    State('project-data-store', 'data')
)
def display_page(pathname, project_data):
    default_report_colormap = 'viridis'
    if project_data and 'colormap' in project_data:
        default_report_colormap = project_data['colormap']

    if pathname == '/processing':
        return processing.layout
    elif pathname == '/report':
        return report.create_report_layout(default_palette_name=default_report_colormap)
    else:
        return processing.layout

# Register callbacks from pages
# Callbacks from 'processing.py' are now registered via a function call
processing.register_callbacks(app) # <--- CORRECTED: Calling the function

# Callbacks from 'report.py'
report.register_report_callbacks(app)


if __name__ == '__main__':
    app.run(debug=True)
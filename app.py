import dash
from dash import Dash, dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.SLATE,
        # "/assets/style.css",
        dbc.icons.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css"
    ]
)
server = app.server

# define navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="/projects/dashboard")),
        dbc.NavItem(dbc.NavLink("Alert", href="/projects/alert"))
    ],
    brand="AI Credit Analysis",
    brand_href="/",
    color='Dark',
    dark=True,
    style= {"border": "none"},
)

# layout
app.layout = dbc.Container(
    children=[
        navbar,
        dash.page_container,
    ],
    fluid=True
)


if __name__ == "__main__":
    app.run(debug=False)

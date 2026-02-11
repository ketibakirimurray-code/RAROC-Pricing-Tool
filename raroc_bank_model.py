# RAROC Model for Commercial Term Loan Pricing
# Phase 1: Cash Flow Calculations and Present Value Analysis

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__)

# PD and LGD mapping tables
PD_SCALE = {
    1: 0.0010, 2: 0.0025, 3: 0.0050, 4: 0.0100, 5: 0.0200,
    6: 0.0400, 7: 0.0800, 8: 0.1500, 9: 0.2500, 10: 0.4000,
    11: 0.6000, 12: 0.8000, 13: 0.9500
}

LGD_SCALE = {
    'A': 0.10, 'B': 0.20, 'C': 0.30, 'D': 0.40,
    'E': 0.50, 'F': 0.60, 'G': 0.75, 'H': 0.90
}

def calculate_monthly_payment(principal, annual_rate, term_months):
    """Calculate monthly P&I payment"""
    if annual_rate == 0:
        return principal / term_months
    monthly_rate = annual_rate / 12 / 100
    payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
              ((1 + monthly_rate)**term_months - 1)
    return payment

def generate_amortization_schedule(principal, annual_rate, term_months, ftp_rate,
                                   nii_fee, nii_months, nie_amount, discount_rate):
    """Generate complete amortization schedule with all cash flows"""

    monthly_payment = calculate_monthly_payment(principal, annual_rate, term_months)
    monthly_rate = annual_rate / 12 / 100
    monthly_ftp_rate = ftp_rate / 12 / 100
    monthly_discount_rate = discount_rate / 12 / 100

    schedule = []
    balance = principal

    for month in range(1, term_months + 1):
        # Interest and principal breakdown
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance = max(0, balance - principal_payment)

        # Interest Income = Interest Payment
        interest_income = interest_payment

        # Interest Expense = FTP rate on beginning balance
        interest_expense = (balance + principal_payment) * monthly_ftp_rate

        # Non-Interest Income
        non_interest_income = nii_fee if month <= nii_months else 0

        # Non-Interest Expense
        non_interest_expense = nie_amount

        # Present Value calculations
        discount_factor = 1 / ((1 + monthly_discount_rate) ** month)
        pv_interest_income = interest_income * discount_factor
        pv_interest_expense = interest_expense * discount_factor
        pv_non_interest_income = non_interest_income * discount_factor
        pv_non_interest_expense = non_interest_expense * discount_factor

        # Net Income
        net_income = (interest_income - interest_expense +
                     non_interest_income - non_interest_expense)
        pv_net_income = net_income * discount_factor

        schedule.append({
            'Month': month,
            'Beginning_Balance': balance + principal_payment,
            'Payment': monthly_payment,
            'Principal': principal_payment,
            'Interest': interest_payment,
            'Ending_Balance': balance,
            'Interest_Income': interest_income,
            'Interest_Expense': interest_expense,
            'Non_Interest_Income': non_interest_income,
            'Non_Interest_Expense': non_interest_expense,
            'Net_Income': net_income,
            'PV_Interest_Income': pv_interest_income,
            'PV_Interest_Expense': pv_interest_expense,
            'PV_Non_Interest_Income': pv_non_interest_income,
            'PV_Non_Interest_Expense': pv_non_interest_expense,
            'PV_Net_Income': pv_net_income,
            'Discount_Factor': discount_factor
        })

    return pd.DataFrame(schedule)

def calculate_summary_metrics(df):
    """Calculate summary metrics from amortization schedule"""
    return {
        'Total_Interest_Income': df['Interest_Income'].sum(),
        'Total_Interest_Expense': df['Interest_Expense'].sum(),
        'Total_Non_Interest_Income': df['Non_Interest_Income'].sum(),
        'Total_Non_Interest_Expense': df['Non_Interest_Expense'].sum(),
        'Total_Net_Income': df['Net_Income'].sum(),
        'PV_Interest_Income': df['PV_Interest_Income'].sum(),
        'PV_Interest_Expense': df['PV_Interest_Expense'].sum(),
        'PV_Non_Interest_Income': df['PV_Non_Interest_Income'].sum(),
        'PV_Non_Interest_Expense': df['PV_Non_Interest_Expense'].sum(),
        'PV_Net_Income': df['PV_Net_Income'].sum()
    }

# App Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('🏦 RAROC Model - Commercial Term Loan Pricing',
                style={'color': 'white', 'textAlign': 'center', 'margin': '0', 'padding': '20px'}),
        html.P('Phase 1: Cash Flow Analysis & Present Value Calculations',
               style={'color': 'white', 'textAlign': 'center', 'margin': '0', 'paddingBottom': '20px'})
    ], style={'backgroundColor': '#2c3e50'}),

    # Main Content
    html.Div([
        # Input Method Selection
        html.Div([
            html.H3('Select Input Method'),
            dcc.RadioItems(
                id='input-method',
                options=[
                    {'label': ' Manual Entry', 'value': 'manual'},
                    {'label': ' Upload File (CSV/Excel)', 'value': 'file'}
                ],
                value='manual',
                style={'fontSize': '16px'}
            )
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px', 'borderRadius': '10px'}),

        # Manual Input Section
        html.Div(id='manual-input-section', children=[
            html.H3('Loan Parameters'),
            html.Div([
                # Column 1
                html.Div([
                    html.Label('Original Balance ($):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='principal', type='number', value=1000000,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                    html.Label('Annual Interest Rate (%):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='interest-rate', type='number', value=6.5, step=0.01,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                    html.Label('Term (Months):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='term', type='number', value=100,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                    html.Label('FTP Cost (%):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='ftp-rate', type='number', value=2.3, step=0.01,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

                # Column 2
                html.Div([
                    html.Label('Discount Rate (%):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='discount-rate', type='number', value=2.5, step=0.01,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                    html.Label('Non-Interest Income ($/month):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='nii-fee', type='number', value=100,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                    html.Label('NII Collection Period (Months):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='nii-months', type='number', value=50,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                    html.Label('Non-Interest Expense ($/month):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='nie-amount', type='number', value=200,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

                # Column 3
                html.Div([
                    html.Label('PD Rating (1-13):', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='pd-rating',
                        options=[{'label': f'Rating {i} ({PD_SCALE[i]:.2%})', 'value': i}
                                for i in range(1, 14)],
                        value=5,
                        style={'marginBottom': '15px'}
                    ),

                    html.Label('LGD Grade (A-H):', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='lgd-grade',
                        options=[{'label': f'Grade {k} ({v:.0%})', 'value': k}
                                for k, v in LGD_SCALE.items()],
                        value='C',
                        style={'marginBottom': '15px'}
                    ),

                    html.Label('Zip Code:', style={'fontWeight': 'bold'}),
                    dcc.Input(id='zip-code', type='text', value='45208',
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                    html.Label('Loan ID:', style={'fontWeight': 'bold'}),
                    dcc.Input(id='loan-id', type='text', value='LOAN-001',
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            ]),

            html.Div([
                html.Button('Calculate Cash Flows', id='calculate-btn', n_clicks=0,
                           style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '12px 30px',
                                  'border': 'none', 'borderRadius': '5px', 'fontSize': '16px',
                                  'cursor': 'pointer', 'marginTop': '20px'})
            ], style={'textAlign': 'center'})
        ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
                 'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)'}),

        # File Upload Section
        html.Div(id='file-upload-section', children=[
            html.H3('Upload Loan Data File'),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a CSV/Excel File', style={'color': '#3498db', 'cursor': 'pointer'})
                ]),
                style={
                    'width': '100%', 'height': '80px', 'lineHeight': '80px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center', 'backgroundColor': '#f9f9f9'
                },
                multiple=False
            ),
            html.Div(id='upload-status', style={'marginTop': '10px', 'color': '#27ae60', 'fontWeight': 'bold'})
        ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
                 'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'display': 'none'}),

        # Results Section
        html.Div(id='results-section', children=[
            # Summary Metrics
            html.Div(id='summary-cards'),

            # Visualizations
            html.Div([
                html.H3('Cash Flow Visualizations', style={'textAlign': 'center', 'marginTop': '30px'}),
                dcc.Graph(id='cashflow-chart'),
                dcc.Graph(id='balance-chart'),
                dcc.Graph(id='pv-comparison-chart'),
            ]),

            # Detailed Table
            html.Div([
                html.H3('Detailed Amortization Schedule', style={'marginTop': '30px'}),
                html.Div([
                    html.Button('Download Full Schedule (CSV)', id='download-btn',
                               style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                                      'border': 'none', 'borderRadius': '5px', 'marginBottom': '10px'}),
                    dcc.Download(id='download-dataframe-csv'),
                ]),
                html.Div(id='amortization-table', style={'overflowX': 'auto'})
            ])
        ], style={'marginTop': '20px'})

    ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': 'auto'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f5f5', 'minHeight': '100vh'})

# Store for the dataframe
@app.callback(
    [Output('manual-input-section', 'style'),
     Output('file-upload-section', 'style')],
    Input('input-method', 'value')
)
def toggle_input_method(method):
    if method == 'manual':
        return {'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
                'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'display': 'block'}, \
               {'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
                'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'display': 'none'}
    else:
        return {'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
                'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'display': 'none'}, \
               {'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
                'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'display': 'block'}

# Main calculation callback
@app.callback(
    [Output('summary-cards', 'children'),
     Output('cashflow-chart', 'figure'),
     Output('balance-chart', 'figure'),
     Output('pv-comparison-chart', 'figure'),
     Output('amortization-table', 'children')],
    [Input('calculate-btn', 'n_clicks'),
     Input('upload-data', 'contents')],
    [State('principal', 'value'),
     State('interest-rate', 'value'),
     State('term', 'value'),
     State('ftp-rate', 'value'),
     State('discount-rate', 'value'),
     State('nii-fee', 'value'),
     State('nii-months', 'value'),
     State('nie-amount', 'value'),
     State('pd-rating', 'value'),
     State('lgd-grade', 'value'),
     State('zip-code', 'value'),
     State('loan-id', 'value'),
     State('upload-data', 'filename')]
)
def update_results(n_clicks, contents, principal, interest_rate, term, ftp_rate,
                  discount_rate, nii_fee, nii_months, nie_amount, pd_rating,
                  lgd_grade, zip_code, loan_id, filename):

    if n_clicks == 0 and contents is None:
        return [html.Div()], {}, {}, {}, html.Div()

    # Generate amortization schedule
    df = generate_amortization_schedule(
        principal, interest_rate, term, ftp_rate,
        nii_fee, nii_months, nie_amount, discount_rate
    )

    # Calculate summary metrics
    metrics = calculate_summary_metrics(df)

    # Summary Cards
    summary_cards = html.Div([
        html.H3('Summary Metrics', style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            # Row 1 - Totals
            html.Div([
                html.Div([
                    html.H4('Total Interest Income', style={'color': '#27ae60', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['Total_Interest_Income']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#e8f8f5', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('Total Interest Expense', style={'color': '#e74c3c', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['Total_Interest_Expense']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#fadbd8', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('Total NII', style={'color': '#3498db', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['Total_Non_Interest_Income']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#d6eaf8', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('Total NIE', style={'color': '#e67e22', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['Total_Non_Interest_Expense']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#fdebd0', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),

            # Row 2 - Present Values
            html.Div([
                html.Div([
                    html.H4('PV Interest Income', style={'color': '#27ae60', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['PV_Interest_Income']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#e8f8f5', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('PV Interest Expense', style={'color': '#e74c3c', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['PV_Interest_Expense']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#fadbd8', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('PV NII', style={'color': '#3498db', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['PV_Non_Interest_Income']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#d6eaf8', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('PV NIE', style={'color': '#e67e22', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['PV_Non_Interest_Expense']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#fdebd0', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),

            # Row 3 - Net Income
            html.Div([
                html.Div([
                    html.H4('Total Net Income', style={'color': '#9b59b6', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['Total_Net_Income']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#ebdef0', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('PV Net Income', style={'color': '#9b59b6', 'marginBottom': '5px'}),
                    html.H2(f"${metrics['PV_Net_Income']:,.2f}", style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#ebdef0', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),

                html.Div([
                    html.H4('Monthly Payment', style={'color': '#34495e', 'marginBottom': '5px'}),
                    html.H2(f"${calculate_monthly_payment(principal, interest_rate, term):,.2f}",
                           style={'margin': '0'})
                ], className='metric-card', style={'backgroundColor': '#ecf0f1', 'padding': '20px',
                                                    'borderRadius': '10px', 'textAlign': 'center',
                                                    'flex': '1', 'margin': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
             'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'})

    # Cash Flow Chart
    cashflow_fig = go.Figure()
    cashflow_fig.add_trace(go.Scatter(x=df['Month'], y=df['Interest_Income'],
                                      name='Interest Income', line=dict(color='#27ae60', width=2)))
    cashflow_fig.add_trace(go.Scatter(x=df['Month'], y=df['Interest_Expense'],
                                      name='Interest Expense', line=dict(color='#e74c3c', width=2)))
    cashflow_fig.add_trace(go.Scatter(x=df['Month'], y=df['Non_Interest_Income'],
                                      name='Non-Interest Income', line=dict(color='#3498db', width=2)))
    cashflow_fig.add_trace(go.Scatter(x=df['Month'], y=df['Non_Interest_Expense'],
                                      name='Non-Interest Expense', line=dict(color='#e67e22', width=2)))
    cashflow_fig.add_trace(go.Scatter(x=df['Month'], y=df['Net_Income'],
                                      name='Net Income', line=dict(color='#9b59b6', width=3, dash='dash')))

    cashflow_fig.update_layout(
        title='Monthly Cash Flows Over Loan Term',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        height=500
    )

    # Balance Chart
    balance_fig = go.Figure()
    balance_fig.add_trace(go.Scatter(x=df['Month'], y=df['Beginning_Balance'],
                                     name='Outstanding Balance', fill='tozeroy',
                                     line=dict(color='#2980b9', width=3)))

    balance_fig.update_layout(
        title='Loan Outstanding Balance Over Time',
        xaxis_title='Month',
        yaxis_title='Balance ($)',
        hovermode='x unified',
        height=400
    )

    # PV Comparison Chart
    pv_data = pd.DataFrame({
        'Category': ['Interest Income', 'Interest Expense', 'Non-Interest Income', 'Non-Interest Expense'],
        'Total': [metrics['Total_Interest_Income'], metrics['Total_Interest_Expense'],
                 metrics['Total_Non_Interest_Income'], metrics['Total_Non_Interest_Expense']],
        'Present Value': [metrics['PV_Interest_Income'], metrics['PV_Interest_Expense'],
                         metrics['PV_Non_Interest_Income'], metrics['PV_Non_Interest_Expense']]
    })

    pv_fig = go.Figure()
    pv_fig.add_trace(go.Bar(x=pv_data['Category'], y=pv_data['Total'],
                           name='Total (Nominal)', marker_color='#95a5a6'))
    pv_fig.add_trace(go.Bar(x=pv_data['Category'], y=pv_data['Present Value'],
                           name='Present Value', marker_color='#3498db'))

    pv_fig.update_layout(
        title='Total vs Present Value Comparison',
        xaxis_title='Category',
        yaxis_title='Amount ($)',
        barmode='group',
        height=400
    )

    # Amortization Table (showing first 24 months + last 6 months)
    display_df = pd.concat([df.head(24), df.tail(6)])
    display_df_formatted = display_df.copy()

    # Format currency columns
    currency_cols = ['Beginning_Balance', 'Payment', 'Principal', 'Interest', 'Ending_Balance',
                    'Interest_Income', 'Interest_Expense', 'Non_Interest_Income',
                    'Non_Interest_Expense', 'Net_Income', 'PV_Interest_Income',
                    'PV_Interest_Expense', 'PV_Non_Interest_Income', 'PV_Non_Interest_Expense',
                    'PV_Net_Income']

    for col in currency_cols:
        display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f'${x:,.2f}')

    display_df_formatted['Discount_Factor'] = display_df_formatted['Discount_Factor'].apply(lambda x: f'{x:.6f}')

    table = dash_table.DataTable(
        data=display_df_formatted.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in display_df_formatted.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'right',
            'padding': '10px',
            'fontSize': '12px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': '#2c3e50',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            }
        ],
        page_size=15
    )

    table_div = html.Div([
        html.P(f'Showing first 24 months and last 6 months (Total: {len(df)} months)',
               style={'fontStyle': 'italic', 'color': '#7f8c8d', 'marginBottom': '10px'}),
        table
    ])

    return summary_cards, cashflow_fig, balance_fig, pv_fig, table_div

# Download callback
@app.callback(
    Output('download-dataframe-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    [State('principal', 'value'),
     State('interest-rate', 'value'),
     State('term', 'value'),
     State('ftp-rate', 'value'),
     State('discount-rate', 'value'),
     State('nii-fee', 'value'),
     State('nii-months', 'value'),
     State('nie-amount', 'value')],
    prevent_initial_call=True
)
def download_schedule(n_clicks, principal, interest_rate, term, ftp_rate,
                     discount_rate, nii_fee, nii_months, nie_amount):
    df = generate_amortization_schedule(
        principal, interest_rate, term, ftp_rate,
        nii_fee, nii_months, nie_amount, discount_rate
    )
    return dcc.send_data_frame(df.to_csv, "amortization_schedule.csv", index=False)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("RAROC Model - Commercial Term Loan Pricing")
    print("="*70)
    print("\nPhase 1: Cash Flow Analysis")
    print("  - Calculate Interest Income & Expense")
    print("  - Calculate Non-Interest Income & Expense")
    print("  - Present Value calculations")
    print("  - Amortization schedule generation")
    print("\nOpen your browser: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    app.run(debug=True, port=8050)

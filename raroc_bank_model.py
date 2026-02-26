# RAROC Model for Commercial Term Loan Pricing
# Phase 1 & 2: Cash Flows, EL, Economic Capital, Regulatory Capital, RAROC

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import norm
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

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_expected_loss(df, pd_rate, lgd_rate):
    """
    Expected Loss (EL) per month = PD × LGD × EAD
    EAD = Beginning Balance for that month
    Monthly PD = 1 - (1 - Annual PD)^(1/12)
    """
    monthly_pd = 1 - (1 - pd_rate) ** (1 / 12)
    df = df.copy()
    df['Monthly_PD']  = monthly_pd
    df['LGD']         = lgd_rate
    df['EAD']         = df['Beginning_Balance']
    df['Monthly_EL']  = monthly_pd * lgd_rate * df['EAD']
    df['Cumulative_EL'] = df['Monthly_EL'].cumsum()
    return df

def calculate_economic_capital(pd_rate, lgd_rate, ead, maturity_years=8.33,
                                confidence=0.999):
    """
    Basel II IRB (Vasicek model) Economic Capital
    ─────────────────────────────────────────────
    R   = Asset correlation (Basel formula)
    b   = Maturity adjustment exponent
    MA  = Maturity adjustment factor
    K   = Capital requirement %
    EC  = K × EAD  (Unexpected Loss capital buffer)
    """
    if pd_rate <= 0:
        pd_rate = 0.0001
    if pd_rate >= 1:
        pd_rate = 0.9999

    # Asset correlation (Basel II corporate formula)
    R = (0.12 * (1 - np.exp(-50 * pd_rate)) / (1 - np.exp(-50)) +
         0.24 * (1 - (1 - np.exp(-50 * pd_rate)) / (1 - np.exp(-50))))

    # Maturity adjustment
    b = (0.11852 - 0.05478 * np.log(pd_rate)) ** 2
    MA = (1 + (maturity_years - 2.5) * b) / (1 - 1.5 * b)

    # Capital requirement (K)
    N    = norm.cdf
    Ninv = norm.ppf
    K = (lgd_rate * N((1 - R) ** (-0.5) * Ninv(pd_rate) +
                       (R / (1 - R)) ** 0.5 * Ninv(confidence)) -
         pd_rate * lgd_rate) * MA

    K = max(K, 0)
    ec  = K * ead
    el  = pd_rate * lgd_rate * ead
    uel = ec  # EC already nets out EL in Basel formula

    return {
        'K_pct':            K,
        'Economic_Capital':  ec,
        'Asset_Correlation': R,
        'Maturity_Adj':      MA,
        'UEL':               uel,
        'EL':                el
    }

def calculate_regulatory_capital(pd_rate, lgd_rate, ead, loan_type='commercial'):
    """
    Regulatory Capital (Basel III Standardised Approach)
    ────────────────────────────────────────────────────
    Risk Weight (RW) is mapped from PD bucket.
    Reg C = EAD × RW × 8% (minimum CET1 ratio)
    """
    # Risk weight schedule (PD → RW for commercial/corporate loans)
    if   pd_rate <= 0.0025: rw = 0.20
    elif pd_rate <= 0.005:  rw = 0.35
    elif pd_rate <= 0.01:   rw = 0.50
    elif pd_rate <= 0.02:   rw = 0.75
    elif pd_rate <= 0.04:   rw = 1.00
    elif pd_rate <= 0.08:   rw = 1.50
    elif pd_rate <= 0.15:   rw = 2.00
    elif pd_rate <= 0.25:   rw = 2.50
    elif pd_rate <= 0.40:   rw = 3.00
    elif pd_rate <= 0.60:   rw = 3.50
    else:                   rw = 4.00

    min_capital_ratio = 0.08   # 8% minimum Basel III
    tier1_ratio       = 0.10   # 10% Tier 1 target (common for community banks)

    rwa      = ead * rw
    reg_cap  = rwa * min_capital_ratio
    tier1    = rwa * tier1_ratio

    return {
        'Risk_Weight':       rw,
        'RWA':               rwa,
        'Reg_Capital':       reg_cap,
        'Tier1_Capital':     tier1,
        'Min_Capital_Ratio': min_capital_ratio,
        'Tier1_Ratio':       tier1_ratio
    }

def calculate_raroc(net_income, el, economic_capital):
    """
    RAROC = (Net Income – Expected Loss) / Economic Capital
    A RAROC > Hurdle Rate (typically 10-15%) = value-creating loan
    """
    if economic_capital <= 0:
        return 0
    risk_adjusted_return = net_income - el
    raroc = risk_adjusted_return / economic_capital
    return raroc

# App Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('🏦 RAROC Model - Commercial Term Loan Pricing',
                style={'color': 'white', 'textAlign': 'center', 'margin': '0', 'padding': '20px'}),
        html.P('Phase 1: Cash Flows & PV  |  Phase 2: EL, Economic Capital, Reg Capital & RAROC',
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

                    html.Label('Hurdle Rate (%):', style={'fontWeight': 'bold'}),
                    dcc.Input(id='hurdle-rate', type='number', value=12.0, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '5px'}),
                    html.P('Min RAROC for value-creating loan',
                           style={'fontSize': '11px', 'color': '#7f8c8d', 'marginBottom': '15px'}),
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
                dcc.Graph(id='income-chart'),
                dcc.Graph(id='expense-chart'),
                dcc.Graph(id='netincome-chart'),
                dcc.Graph(id='balance-chart'),
            ]),

            # Detailed Table
            html.Div([
                html.H3('Detailed Amortization Schedule', style={'marginTop': '30px'}),
                html.Div([
                    html.Button('Download Full Schedule (CSV)', id='download-btn',
                               style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                                      'border': 'none', 'borderRadius': '5px', 'marginBottom': '10px',
                                      'marginRight': '10px'}),
                    html.Button('📅 View All Monthly Cash Flows', id='toggle-monthly-btn', n_clicks=0,
                               style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '10px 20px',
                                      'border': 'none', 'borderRadius': '5px', 'marginBottom': '10px'}),
                    dcc.Download(id='download-dataframe-csv'),
                ]),
                html.Div(id='amortization-table', style={'overflowX': 'auto'}),

                # Collapsible Monthly Cash Flow Section
                html.Div(id='monthly-cashflow-section', children=[
                    html.Hr(style={'marginTop': '30px'}),
                    html.H3('📊 Monthly Cash Flow Detail', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.Div(id='monthly-cashflow-table', style={'overflowX': 'auto', 'marginTop': '20px'})
                ], style={'display': 'none'})
            ])
        ], style={'marginTop': '20px'}),

        # ── PHASE 2 RESULTS ──────────────────────────────────────────────────
        html.Div(id='phase2-section', style={'marginTop': '40px', 'display': 'none'}, children=[
            html.Hr(),
            html.Div([
                html.H2('Phase 2: Credit Risk & Capital',
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
                html.P('Expected Loss  |  Economic Capital  |  Regulatory Capital  |  RAROC',
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px',
                              'marginBottom': '30px'})
            ]),

            # RAROC verdict banner
            html.Div(id='raroc-banner'),

            # Phase 2 KPI cards
            html.Div(id='phase2-cards'),

            # Phase 2 charts
            html.Div([
                dcc.Graph(id='el-chart'),
                dcc.Graph(id='capital-chart'),
                dcc.Graph(id='raroc-waterfall'),
            ]),

            # Monthly EL table (toggleable)
            html.Div([
                html.Button('View Monthly EL Schedule', id='toggle-el-btn', n_clicks=0,
                           style={'backgroundColor': '#8e44ad', 'color': 'white',
                                  'padding': '10px 20px', 'border': 'none',
                                  'borderRadius': '5px', 'marginBottom': '10px',
                                  'cursor': 'pointer'}),
                html.Div(id='el-table-section', style={'display': 'none'},
                         children=[html.Div(id='el-monthly-table')])
            ])
        ])

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
     Output('income-chart', 'figure'),
     Output('expense-chart', 'figure'),
     Output('netincome-chart', 'figure'),
     Output('balance-chart', 'figure'),
     Output('amortization-table', 'children'),
     Output('phase2-section', 'style'),
     Output('raroc-banner', 'children'),
     Output('phase2-cards', 'children'),
     Output('el-chart', 'figure'),
     Output('capital-chart', 'figure'),
     Output('raroc-waterfall', 'figure')],
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
     State('hurdle-rate', 'value'),
     State('upload-data', 'filename')]
)
def update_results(n_clicks, contents, principal, interest_rate, term, ftp_rate,
                  discount_rate, nii_fee, nii_months, nie_amount, pd_rating,
                  lgd_grade, zip_code, loan_id, hurdle_rate, filename):

    empty = [{}, {}, {}, {}, {}, {}]
    if n_clicks == 0 and contents is None:
        return [html.Div()], {}, {}, {}, {}, html.Div(), \
               {'display': 'none'}, html.Div(), html.Div(), {}, {}, {}

    # Generate amortization schedule
    df = generate_amortization_schedule(
        principal, interest_rate, term, ftp_rate,
        nii_fee, nii_months, nie_amount, discount_rate
    )

    # Calculate summary metrics
    metrics = calculate_summary_metrics(df)

    # Tax calculation
    tax_rate   = (hurdle_rate or 12.0) * 0   # placeholder — pulled from input below
    tax_rate   = 0.21                          # 21% federal corporate tax rate (no input yet — see note)
    after_tax  = lambda x: x * (1 - tax_rate)

    # Helper: card with hover tooltip
    def p1_card(title, value, tooltip, bg, fg):
        return html.Div([
            html.Div([
                html.Span(title, style={'fontWeight': 'bold', 'color': fg, 'fontSize': '13px'}),
                html.Span(' ⓘ', title=tooltip,
                          style={'cursor': 'help', 'color': '#aaa', 'fontSize': '13px',
                                 'marginLeft': '4px'})
            ], style={'marginBottom': '8px'}),
            html.H2(value, style={'margin': '0', 'fontSize': '20px'})
        ], style={'backgroundColor': bg, 'padding': '18px', 'borderRadius': '10px',
                  'textAlign': 'center', 'flex': '1', 'margin': '8px'})

    summary_cards = html.Div([
        html.H3('Phase 1 — Summary Metrics  (hover ⓘ for formula)',
                style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'}),
        html.P('* Tax not included in Phase 1 totals. Pre-tax figures shown. '
               'After-tax figures appear in the RAROC section below.',
               style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '12px',
                      'marginBottom': '20px', 'fontStyle': 'italic'}),
        html.Div([
            # Row 1 — Totals (nominal)
            html.Div([
                p1_card('Total Interest Income',
                        f"${metrics['Total_Interest_Income']:,.2f}",
                        'Formula: Sum of (Outstanding Balance × Loan Rate / 12) for each month.\n'
                        'This is the gross revenue the bank earns from charging the borrower interest.',
                        '#e8f8f5', '#27ae60'),
                p1_card('Total Interest Expense',
                        f"${metrics['Total_Interest_Expense']:,.2f}",
                        'Formula: Sum of (Outstanding Balance × FTP Rate / 12) for each month.\n'
                        'FTP (Funds Transfer Pricing) is the internal cost the bank pays to fund the loan.',
                        '#fadbd8', '#e74c3c'),
                p1_card('Total Non-Interest Income',
                        f"${metrics['Total_Non_Interest_Income']:,.2f}",
                        'Formula: Fixed fee × number of collection months.\n'
                        'Fees, service charges, or other non-interest revenue tied to the loan.',
                        '#d6eaf8', '#2980b9'),
                p1_card('Total Non-Interest Expense',
                        f"${metrics['Total_Non_Interest_Expense']:,.2f}",
                        'Formula: Fixed operating cost × term months.\n'
                        'Direct costs to originate, service, and administer the loan (salaries, systems, etc.).',
                        '#fdebd0', '#e67e22'),
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

            # Row 2 — Present Values
            html.Div([
                p1_card('PV Interest Income',
                        f"${metrics['PV_Interest_Income']:,.2f}",
                        'Formula: Interest Income(t) / (1 + Discount Rate/12)^t  summed over all months.\n'
                        'Present value tells you what future interest income is worth in today\'s dollars.',
                        '#e8f8f5', '#27ae60'),
                p1_card('PV Interest Expense',
                        f"${metrics['PV_Interest_Expense']:,.2f}",
                        'Formula: Interest Expense(t) / (1 + Discount Rate/12)^t  summed over all months.\n'
                        'Present value of the FTP funding cost discounted to today.',
                        '#fadbd8', '#e74c3c'),
                p1_card('PV Non-Interest Income',
                        f"${metrics['PV_Non_Interest_Income']:,.2f}",
                        'Formula: NII Fee(t) / (1 + Discount Rate/12)^t  summed over collection months.\n'
                        'Present value of all fee income received during the collection period.',
                        '#d6eaf8', '#2980b9'),
                p1_card('PV Non-Interest Expense',
                        f"${metrics['PV_Non_Interest_Expense']:,.2f}",
                        'Formula: NIE(t) / (1 + Discount Rate/12)^t  summed over all months.\n'
                        'Present value of all operating costs over the life of the loan.',
                        '#fdebd0', '#e67e22'),
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

            # Row 3 — Net Income
            html.Div([
                p1_card('Total Net Income (Pre-Tax)',
                        f"${metrics['Total_Net_Income']:,.2f}",
                        'Formula: Interest Income - Interest Expense + NII - NIE  (summed over all months).\n'
                        'Pre-tax profit generated by the loan. TAX IS NOT INCLUDED HERE.',
                        '#ebdef0', '#8e44ad'),
                p1_card('PV Net Income (Pre-Tax)',
                        f"${metrics['PV_Net_Income']:,.2f}",
                        'Formula: Net Income(t) / (1 + Discount Rate/12)^t  summed over all months.\n'
                        'The present value of all future pre-tax profits. Used as the numerator in RAROC.',
                        '#ebdef0', '#8e44ad'),
                p1_card('After-Tax Net Income (est.)',
                        f"${after_tax(metrics['Total_Net_Income']):,.2f}",
                        f'Formula: Pre-Tax Net Income × (1 - Tax Rate).\n'
                        f'Assumes 21% federal corporate tax rate. '
                        f'After-tax income = ${metrics["Total_Net_Income"]:,.2f} × (1 - 0.21).',
                        '#d5f5e3', '#1e8449'),
                p1_card('Monthly Payment',
                        f"${calculate_monthly_payment(principal, interest_rate, term):,.2f}",
                        'Formula: P × [r(1+r)^n] / [(1+r)^n - 1]\n'
                        'where P = principal, r = monthly rate, n = term months.\n'
                        'Fixed P&I payment the borrower makes every month.',
                        '#ecf0f1', '#34495e'),
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
             'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'})

    # Chart 1: Income Stacked Area Chart
    income_fig = go.Figure()

    # Stack the income components
    income_fig.add_trace(go.Scatter(
        x=df['Month'], y=df['Interest_Income'],
        name='Interest Income',
        mode='lines',
        line=dict(width=0.5, color='#27ae60'),
        stackgroup='income',
        fillcolor='rgba(39, 174, 96, 0.7)'
    ))

    income_fig.add_trace(go.Scatter(
        x=df['Month'], y=df['Non_Interest_Income'],
        name='Non-Interest Income',
        mode='lines',
        line=dict(width=0.5, color='#3498db'),
        stackgroup='income',
        fillcolor='rgba(52, 152, 219, 0.7)'
    ))

    income_fig.update_layout(
        title='💰 Total Income Breakdown (Stacked)',
        xaxis_title='Month',
        yaxis_title='Income ($)',
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Chart 2: Expense Stacked Area Chart
    expense_fig = go.Figure()

    # Stack the expense components
    expense_fig.add_trace(go.Scatter(
        x=df['Month'], y=df['Interest_Expense'],
        name='Interest Expense (FTP)',
        mode='lines',
        line=dict(width=0.5, color='#e74c3c'),
        stackgroup='expense',
        fillcolor='rgba(231, 76, 60, 0.7)'
    ))

    expense_fig.add_trace(go.Scatter(
        x=df['Month'], y=df['Non_Interest_Expense'],
        name='Non-Interest Expense',
        mode='lines',
        line=dict(width=0.5, color='#e67e22'),
        stackgroup='expense',
        fillcolor='rgba(230, 126, 34, 0.7)'
    ))

    expense_fig.update_layout(
        title='💸 Total Expenses Breakdown (Stacked)',
        xaxis_title='Month',
        yaxis_title='Expenses ($)',
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Chart 3: Net Income Line Chart
    netincome_fig = go.Figure()

    netincome_fig.add_trace(go.Scatter(
        x=df['Month'], y=df['Net_Income'],
        name='Net Income',
        mode='lines',
        line=dict(color='#9b59b6', width=3),
        fill='tozeroy',
        fillcolor='rgba(155, 89, 182, 0.3)'
    ))

    # Add zero line for reference
    netincome_fig.add_hline(y=0, line_dash="dash", line_color="gray",
                           annotation_text="Break Even", annotation_position="right")

    netincome_fig.update_layout(
        title='📊 Net Income Over Time',
        xaxis_title='Month',
        yaxis_title='Net Income ($)',
        hovermode='x unified',
        height=400,
        showlegend=True
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

    # ── PHASE 2 ──────────────────────────────────────────────────────────────
    pd_rate  = PD_SCALE[pd_rating]
    lgd_rate = LGD_SCALE[lgd_grade]
    ead      = principal                       # EAD = original balance at origination
    hurdle   = (hurdle_rate or 12.0) / 100.0

    # Expected Loss
    df_el     = calculate_expected_loss(df, pd_rate, lgd_rate)
    total_el  = df_el['Monthly_EL'].sum()

    # Economic Capital (on original balance / peak EAD)
    ec_result  = calculate_economic_capital(pd_rate, lgd_rate, ead,
                                            maturity_years=term / 12)
    ec         = ec_result['Economic_Capital']

    # Regulatory Capital
    rc_result  = calculate_regulatory_capital(pd_rate, lgd_rate, ead)
    reg_cap    = rc_result['Reg_Capital']

    # RAROC
    total_net  = metrics['PV_Net_Income']
    raroc_val  = calculate_raroc(total_net, total_el, ec)
    beats_hurdle = raroc_val >= hurdle

    # ── RAROC Verdict Banner ──────────────────────────────────────────────────
    raroc_banner = html.Div([
        html.H2(f'RAROC: {raroc_val:.2%}',
                style={'margin': '0', 'fontSize': '32px', 'fontWeight': 'bold'}),
        html.P(f'Hurdle Rate: {hurdle:.2%}  |  '
               f'{"PASS - This loan CREATES value" if beats_hurdle else "FAIL - This loan DESTROYS value"}',
               style={'margin': '5px 0 0 0', 'fontSize': '16px'})
    ], style={
        'backgroundColor': '#d5f5e3' if beats_hurdle else '#fadbd8',
        'border': f'3px solid {"#27ae60" if beats_hurdle else "#e74c3c"}',
        'borderRadius': '12px', 'padding': '25px',
        'textAlign': 'center', 'marginBottom': '30px',
        'color': '#27ae60' if beats_hurdle else '#e74c3c'
    })

    # After-tax net income for RAROC
    after_tax_net  = metrics['PV_Net_Income'] * (1 - tax_rate)
    raroc_after_tax = calculate_raroc(after_tax_net, total_el, ec)

    # ── Phase 2 KPI Cards ─────────────────────────────────────────────────────
    def kpi(title, value, tooltip, bg, fg):
        return html.Div([
            html.Div([
                html.Span(title, style={'color': fg, 'fontSize': '12px', 'fontWeight': 'bold'}),
                html.Span(' ⓘ', title=tooltip,
                          style={'cursor': 'help', 'color': '#bbb', 'fontSize': '12px',
                                 'marginLeft': '3px'})
            ], style={'marginBottom': '6px'}),
            html.H2(value, style={'margin': '0', 'fontSize': '20px'})
        ], style={'backgroundColor': bg, 'padding': '18px', 'borderRadius': '10px',
                  'textAlign': 'center', 'flex': '1', 'margin': '8px'})

    phase2_cards = html.Div([
        html.H3('Phase 2 — Credit Risk & Capital  (hover ⓘ for formula)',
                style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'}),

        # Row 1 — Risk inputs & EL
        html.Div([
            kpi('PD (Annual)', f'{pd_rate:.2%}',
                'Probability of Default — annual likelihood the borrower will not repay.\n'
                f'PD Rating {pd_rating} maps to {pd_rate:.2%} annual default probability.',
                '#fef9e7', '#d4ac0d'),
            kpi('LGD', f'{lgd_rate:.0%}',
                'Loss Given Default — % of the outstanding balance the bank loses if the borrower defaults.\n'
                f'LGD Grade {lgd_grade} = {lgd_rate:.0%}. Reflects collateral quality and recovery rates.',
                '#fef9e7', '#d4ac0d'),
            kpi('EAD', f'${ead:,.0f}',
                'Exposure at Default — total amount the bank is exposed to at time of default.\n'
                'Set to the original loan balance. In practice this can vary by month.',
                '#fef9e7', '#d4ac0d'),
            kpi('Expected Loss (EL)', f'${total_el:,.2f}',
                'Formula: Monthly EL = Monthly PD × LGD × Beginning Balance\n'
                'Monthly PD = 1 - (1 - Annual PD)^(1/12)\n'
                'Total EL = sum of all monthly ELs over the loan life.\n'
                'EL is the average loss the bank expects to absorb — it should be priced into the loan rate.',
                '#fadbd8', '#e74c3c'),
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),

        # Row 2 — Capital
        html.Div([
            kpi('Economic Capital (EC)', f'${ec:,.2f}',
                'Formula: EC = K × EAD  (Basel II IRB / Vasicek model)\n'
                'K = [LGD × N(G(PD)/√(1-R) + √(R/(1-R)) × G(0.999)) - PD×LGD] × Maturity Adj.\n'
                'EC is the capital buffer needed to absorb UNEXPECTED losses at 99.9% confidence.\n'
                'Unlike EL, EC is not priced in — it is equity set aside by the bank.',
                '#ebdef0', '#8e44ad'),
            kpi('EC as % of EAD', f'{ec_result["K_pct"]:.2%}',
                'K% = Economic Capital / EAD.\n'
                'Represents how much capital per dollar of loan exposure the bank must hold.\n'
                'Higher PD or LGD = higher K%.',
                '#ebdef0', '#8e44ad'),
            kpi('Reg Capital (Basel III)', f'${reg_cap:,.2f}',
                'Formula: Reg Capital = RWA × 8% (minimum CET1 ratio)\n'
                'RWA = EAD × Risk Weight\n'
                'Risk Weight is assigned by PD bucket per Basel III standardised approach.\n'
                'This is the MINIMUM capital regulators require. Banks typically hold more.',
                '#d6eaf8', '#2980b9'),
            kpi('Risk Weight', f'{rc_result["Risk_Weight"]:.0%}',
                'Basel III standardised risk weight assigned based on PD bucket.\n'
                f'PD = {pd_rate:.2%} → Risk Weight = {rc_result["Risk_Weight"]:.0%}.\n'
                'Higher risk weight = more regulatory capital required.',
                '#d6eaf8', '#2980b9'),
            kpi('RWA', f'${rc_result["RWA"]:,.2f}',
                'Risk-Weighted Assets = EAD × Risk Weight.\n'
                'The risk-adjusted measure of loan size used by regulators.\n'
                'Regulatory Capital = RWA × 8%.',
                '#d6eaf8', '#2980b9'),
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),

        # Row 3 — RAROC
        html.Div([
            kpi('PV Net Income (Pre-Tax)', f'${total_net:,.2f}',
                'Formula: Sum of [Net Income(t) / (1 + Discount Rate/12)^t]\n'
                'Net Income = Int Income - Int Expense + NII - NIE\n'
                'This is BEFORE tax and BEFORE subtracting Expected Loss.',
                '#e8f8f5', '#27ae60'),
            kpi('After-Tax PV Net Income', f'${after_tax_net:,.2f}',
                f'Formula: PV Net Income × (1 - Tax Rate)\n'
                f'= ${total_net:,.2f} × (1 - {tax_rate:.0%})\n'
                f'Assumes {tax_rate:.0%} federal corporate tax rate.',
                '#d5f5e3', '#1e8449'),
            kpi('Pre-Tax RAROC', f'{raroc_val:.2%}',
                'Formula: RAROC = (PV Net Income - EL) / Economic Capital\n'
                f'= (${total_net:,.2f} - ${total_el:,.2f}) / ${ec:,.2f}\n'
                'Pre-tax version. Compare to hurdle rate to assess if loan creates value.',
                '#ebdef0' if beats_hurdle else '#fadbd8',
                '#27ae60' if beats_hurdle else '#e74c3c'),
            kpi('After-Tax RAROC', f'{raroc_after_tax:.2%}',
                'Formula: RAROC = (After-Tax PV Net Income - EL) / Economic Capital\n'
                f'= (${after_tax_net:,.2f} - ${total_el:,.2f}) / ${ec:,.2f}\n'
                'After-tax version is preferred for shareholder value analysis.',
                '#d5f5e3' if raroc_after_tax >= hurdle else '#fadbd8',
                '#27ae60' if raroc_after_tax >= hurdle else '#e74c3c'),
            kpi('Hurdle Rate', f'{hurdle:.2%}',
                'The minimum acceptable RAROC for this loan to be considered value-creating.\n'
                'Set by management based on cost of equity / shareholder return expectations.\n'
                'RAROC > Hurdle Rate = loan creates value. RAROC < Hurdle = loan destroys value.',
                '#ecf0f1', '#34495e'),
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
              'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '20px'})

    # ── EL Chart: Monthly EL vs Net Income ───────────────────────────────────
    el_fig = go.Figure()
    el_fig.add_trace(go.Scatter(x=df_el['Month'], y=df_el['Net_Income'],
                                name='Net Income', mode='lines',
                                line=dict(color='#27ae60', width=2)))
    el_fig.add_trace(go.Scatter(x=df_el['Month'], y=df_el['Monthly_EL'],
                                name='Expected Loss (EL)', mode='lines',
                                fill='tozeroy', line=dict(color='#e74c3c', width=2),
                                fillcolor='rgba(231,76,60,0.2)'))
    el_fig.add_trace(go.Scatter(x=df_el['Month'],
                                y=df_el['Net_Income'] - df_el['Monthly_EL'],
                                name='Net Income after EL', mode='lines',
                                line=dict(color='#9b59b6', width=2, dash='dash')))
    el_fig.update_layout(title='Monthly Net Income vs Expected Loss',
                         xaxis_title='Month', yaxis_title='Amount ($)',
                         hovermode='x unified', height=420)

    # ── Capital Chart: EC vs Reg Capital ────────────────────────────────────
    cap_fig = go.Figure()
    cap_fig.add_trace(go.Bar(name='Economic Capital',     x=['Capital'], y=[ec],
                             marker_color='#8e44ad'))
    cap_fig.add_trace(go.Bar(name='Regulatory Capital',   x=['Capital'], y=[reg_cap],
                             marker_color='#2980b9'))
    cap_fig.add_trace(go.Bar(name='Total Expected Loss',  x=['Capital'], y=[total_el],
                             marker_color='#e74c3c'))
    cap_fig.add_trace(go.Bar(name='PV Net Income',        x=['Capital'], y=[max(total_net,0)],
                             marker_color='#27ae60'))
    cap_fig.update_layout(title='Capital Requirements vs Income & Loss',
                          barmode='group', yaxis_title='Amount ($)', height=420)

    # ── RAROC Waterfall ───────────────────────────────────────────────────────
    wf_fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],
        x=['PV Interest\nIncome', '+ PV NII', '- PV Interest\nExpense',
           '- PV NIE', '- Expected\nLoss', 'Risk-Adj\nReturn'],
        y=[metrics['PV_Interest_Income'],
           metrics['PV_Non_Interest_Income'],
           -metrics['PV_Interest_Expense'],
           -metrics['PV_Non_Interest_Expense'],
           -total_el,
           0],
        connector={'line': {'color': 'rgb(63,63,63)'}},
        increasing={'marker': {'color': '#27ae60'}},
        decreasing={'marker': {'color': '#e74c3c'}},
        totals={'marker': {'color': '#2980b9'}}
    ))
    wf_fig.add_hline(y=0, line_dash='dash', line_color='gray')
    wf_fig.update_layout(title='RAROC Waterfall: From Revenue to Risk-Adjusted Return',
                         yaxis_title='Amount ($)', height=450)

    return (summary_cards, income_fig, expense_fig, netincome_fig, balance_fig, table_div,
            {'marginTop': '40px', 'display': 'block'},
            raroc_banner, phase2_cards, el_fig, cap_fig, wf_fig)

# Toggle monthly cash flow section
@app.callback(
    [Output('monthly-cashflow-section', 'style'),
     Output('toggle-monthly-btn', 'children'),
     Output('monthly-cashflow-table', 'children')],
    [Input('toggle-monthly-btn', 'n_clicks'),
     Input('calculate-btn', 'n_clicks')],
    [State('principal', 'value'),
     State('interest-rate', 'value'),
     State('term', 'value'),
     State('ftp-rate', 'value'),
     State('discount-rate', 'value'),
     State('nii-fee', 'value'),
     State('nii-months', 'value'),
     State('nie-amount', 'value')]
)
def toggle_monthly_view(toggle_clicks, calc_clicks, principal, interest_rate, term,
                       ftp_rate, discount_rate, nii_fee, nii_months, nie_amount):

    # If no calculation has been done yet, keep hidden
    if calc_clicks == 0:
        return {'display': 'none'}, '📅 View All Monthly Cash Flows', html.Div()

    # Toggle visibility based on odd/even clicks
    is_visible = (toggle_clicks % 2) == 1

    if is_visible:
        # Generate the full monthly cash flow table
        df = generate_amortization_schedule(
            principal, interest_rate, term, ftp_rate,
            nii_fee, nii_months, nie_amount, discount_rate
        )

        # Create a simplified view focused on cash flows
        monthly_df = df[['Month', 'Beginning_Balance', 'Interest_Income', 'Interest_Expense',
                        'Non_Interest_Income', 'Non_Interest_Expense', 'Net_Income',
                        'PV_Interest_Income', 'PV_Interest_Expense',
                        'PV_Non_Interest_Income', 'PV_Non_Interest_Expense', 'PV_Net_Income']].copy()

        # Format for display
        monthly_df_formatted = monthly_df.copy()
        currency_cols = ['Beginning_Balance', 'Interest_Income', 'Interest_Expense',
                        'Non_Interest_Income', 'Non_Interest_Expense', 'Net_Income',
                        'PV_Interest_Income', 'PV_Interest_Expense',
                        'PV_Non_Interest_Income', 'PV_Non_Interest_Expense', 'PV_Net_Income']

        for col in currency_cols:
            monthly_df_formatted[col] = monthly_df_formatted[col].apply(lambda x: f'${x:,.2f}')

        # Create enhanced table
        monthly_table = dash_table.DataTable(
            data=monthly_df_formatted.to_dict('records'),
            columns=[{'name': col.replace('_', ' ').title(), 'id': col}
                    for col in monthly_df_formatted.columns],
            style_table={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'auto'},
            style_cell={
                'textAlign': 'right',
                'padding': '12px',
                'fontSize': '13px',
                'fontFamily': 'Arial',
                'minWidth': '120px'
            },
            style_header={
                'backgroundColor': '#27ae60',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'position': 'sticky',
                'top': 0
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f0f9ff'
                },
                {
                    'if': {
                        'filter_query': '{Net_Income} contains "-"',
                        'column_id': 'Net_Income'
                    },
                    'backgroundColor': '#fee',
                    'color': '#c00'
                },
                {
                    'if': {'column_id': 'Month'},
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                }
            ],
            fixed_rows={'headers': True},
            page_size=50
        )

        table_container = html.Div([
            html.P(f'📊 Complete Monthly Cash Flow Analysis - All {len(df)} Months',
                  style={'fontWeight': 'bold', 'fontSize': '16px', 'color': '#2c3e50',
                         'textAlign': 'center', 'marginBottom': '15px'}),
            html.P('Scroll to view all months. Red highlights indicate negative net income.',
                  style={'fontStyle': 'italic', 'color': '#7f8c8d', 'textAlign': 'center',
                         'marginBottom': '20px'}),
            monthly_table
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
                 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)'})

        return {'display': 'block'}, '❌ Hide Monthly Cash Flows', table_container

    else:
        return {'display': 'none'}, '📅 View All Monthly Cash Flows', html.Div()

# Toggle monthly EL table
@app.callback(
    [Output('el-table-section', 'style'),
     Output('toggle-el-btn', 'children'),
     Output('el-monthly-table', 'children')],
    [Input('toggle-el-btn', 'n_clicks'),
     Input('calculate-btn', 'n_clicks')],
    [State('principal', 'value'),
     State('interest-rate', 'value'),
     State('term', 'value'),
     State('ftp-rate', 'value'),
     State('discount-rate', 'value'),
     State('nii-fee', 'value'),
     State('nii-months', 'value'),
     State('nie-amount', 'value'),
     State('pd-rating', 'value'),
     State('lgd-grade', 'value')]
)
def toggle_el_table(toggle_clicks, calc_clicks, principal, interest_rate, term,
                    ftp_rate, discount_rate, nii_fee, nii_months, nie_amount,
                    pd_rating, lgd_grade):
    if calc_clicks == 0:
        return {'display': 'none'}, 'View Monthly EL Schedule', html.Div()

    is_visible = (toggle_clicks % 2) == 1
    if not is_visible:
        return {'display': 'none'}, 'View Monthly EL Schedule', html.Div()

    df = generate_amortization_schedule(principal, interest_rate, term, ftp_rate,
                                        nii_fee, nii_months, nie_amount, discount_rate)
    pd_rate  = PD_SCALE[pd_rating]
    lgd_rate = LGD_SCALE[lgd_grade]
    df_el    = calculate_expected_loss(df, pd_rate, lgd_rate)

    display = df_el[['Month', 'Beginning_Balance', 'EAD', 'Monthly_PD', 'LGD',
                      'Monthly_EL', 'Cumulative_EL', 'Net_Income']].copy()
    display['Net_After_EL'] = display['Net_Income'] - display['Monthly_EL']

    for col in ['Beginning_Balance', 'EAD', 'Monthly_EL', 'Cumulative_EL',
                'Net_Income', 'Net_After_EL']:
        display[col] = display[col].apply(lambda x: f'${x:,.2f}')
    display['Monthly_PD'] = display['Monthly_PD'].apply(lambda x: f'{x:.4%}')
    display['LGD']        = display['LGD'].apply(lambda x: f'{x:.0%}')

    col_labels = {'Month': 'Month', 'Beginning_Balance': 'Beg Balance',
                  'EAD': 'EAD', 'Monthly_PD': 'Monthly PD', 'LGD': 'LGD',
                  'Monthly_EL': 'Monthly EL', 'Cumulative_EL': 'Cumulative EL',
                  'Net_Income': 'Net Income', 'Net_After_EL': 'Net After EL'}

    table = dash_table.DataTable(
        data=display.to_dict('records'),
        columns=[{'name': col_labels[c], 'id': c} for c in display.columns],
        style_table={'overflowX': 'auto', 'maxHeight': '500px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'right', 'padding': '10px',
                    'fontSize': '13px', 'fontFamily': 'Arial'},
        style_header={'backgroundColor': '#8e44ad', 'color': 'white',
                      'fontWeight': 'bold', 'textAlign': 'center'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f5eef8'},
            {'if': {'filter_query': '{Net_After_EL} contains "-"',
                    'column_id': 'Net_After_EL'},
             'backgroundColor': '#fadbd8', 'color': '#c0392b', 'fontWeight': 'bold'}
        ],
        fixed_rows={'headers': True},
        page_size=50
    )
    return ({'display': 'block'},
            'Hide Monthly EL Schedule',
            html.Div([
                html.P(f'Monthly Expected Loss Schedule - All {len(display)} Months',
                       style={'fontWeight': 'bold', 'textAlign': 'center',
                              'color': '#2c3e50', 'marginBottom': '10px'}),
                table
            ], style={'backgroundColor': 'white', 'padding': '20px',
                      'borderRadius': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)'}))

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
    print("  - Interest Income & Expense (FTP)")
    print("  - Non-Interest Income & Expense")
    print("  - Present Value calculations")
    print("  - Full amortization schedule")
    print("\nPhase 2: Credit Risk & Capital")
    print("  - Expected Loss (EL = PD x LGD x EAD)")
    print("  - Economic Capital (Basel II IRB / Vasicek)")
    print("  - Regulatory Capital (Basel III Standardised)")
    print("  - RAROC = (Net Income - EL) / Economic Capital")
    print("\nOpen your browser: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    app.run(debug=True, port=8050)

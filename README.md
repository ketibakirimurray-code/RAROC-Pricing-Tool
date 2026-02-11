# 🏦 RAROC Pricing Tool

**Risk-Adjusted Return on Capital (RAROC) Model for Commercial Term Loan Pricing**

A comprehensive banking application for analyzing commercial term loans, calculating cash flows, and determining risk-adjusted returns.

---

## 📊 Overview

This tool helps retail and small banks price commercial term loans by calculating:
- Interest Income and Expense
- Non-Interest Income and Expense
- Present Value of all cash flows
- Expected Loss (EL) - *Coming in Phase 2*
- Economic Capital - *Coming in Phase 2*
- Regulatory Capital - *Coming in Phase 2*
- RAROC calculation - *Coming in Phase 2*

---

## ✨ Features (Phase 1)

### Loan Analysis
- **Full Amortization Schedule:** Complete P&I breakdown for the entire loan term
- **Cash Flow Calculations:**
  - Interest Income (loan revenue)
  - Interest Expense (FTP - Funds Transfer Pricing cost)
  - Non-Interest Income (monthly fees)
  - Non-Interest Expense (operating costs)
- **Present Value Discounting:** All cash flows discounted at specified rate
- **Net Income Analysis:** Period-by-period and total profitability

### Risk Parameters
- **PD Scale (1-13):** Probability of Default ratings with mapped percentages
- **LGD Scale (A-H):** Loss Given Default grades with severity levels
- **Geographic Risk:** Zip code tracking for regional analysis

### User Interface
- **Manual Entry Mode:** Input loan parameters directly in the web interface
- **File Upload Mode:** Upload CSV files with multiple loan records
- **Interactive Visualizations:**
  - Cash flow charts over loan term
  - Balance amortization curves
  - Total vs Present Value comparisons
- **Export Functionality:** Download complete amortization schedules as CSV

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone this repository:**
```bash
git clone https://github.com/YOUR-USERNAME/Pricing-Tool.git
cd Pricing-Tool
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python raroc_bank_model.py
```

4. **Open your browser:**
```
http://127.0.0.1:8050/
```

---

## 📖 Usage Guide

### Manual Entry

1. Select **"Manual Entry"** mode
2. Enter loan parameters:
   - **Original Balance:** Loan amount ($)
   - **Annual Interest Rate:** Fixed rate (%)
   - **Term:** Loan duration (months)
   - **FTP Cost:** Funds Transfer Pricing rate (%)
   - **Discount Rate:** For PV calculations (%)
   - **Non-Interest Income:** Monthly fee amount and collection period
   - **Non-Interest Expense:** Monthly operating costs
   - **PD Rating:** Select from 1-13 scale
   - **LGD Grade:** Select from A-H scale
   - **Zip Code:** Geographic location
   - **Loan ID:** Unique identifier

3. Click **"Calculate Cash Flows"**
4. View results, charts, and detailed tables
5. Download CSV if needed

### File Upload

1. Select **"Upload File (CSV/Excel)"** mode
2. Prepare CSV file with these columns:
   ```
   loan_id, principal, interest_rate, term, ftp_rate, discount_rate,
   nii_fee, nii_months, nie_amount, pd_rating, lgd_grade, zip_code
   ```
3. Drag and drop or select file
4. View aggregated results

**Sample file included:** `sample_loan_data.csv`

---

## 📐 Example Scenario

**Default Loan Parameters:**
- Loan Amount: $1,000,000
- Interest Rate: 6.5% (fixed)
- Term: 100 months
- FTP Cost: 2.3%
- Discount Rate: 2.5%
- NII: $100/month for 50 months
- NIE: $200/month
- Monthly Payment: ~$12,775

**Expected Results:**
- Total Interest Income: ~$277,000
- Total Interest Expense: ~$121,000
- Net Interest Margin: ~$156,000
- PV calculations reflect time value of money

---

## 🔢 Key Formulas

### Monthly Payment (P&I)
```
PMT = P × [r(1+r)^n] / [(1+r)^n - 1]

Where:
  P = Principal (loan amount)
  r = Monthly interest rate (annual rate / 12)
  n = Number of months
```

### Interest Income
```
Interest Income = Outstanding Balance × Monthly Interest Rate
```

### Interest Expense (FTP)
```
Interest Expense = Outstanding Balance × Monthly FTP Rate
```

### Present Value
```
PV = Cash Flow / (1 + Monthly Discount Rate)^month
```

### Net Income
```
Net Income = Interest Income - Interest Expense +
             Non-Interest Income - Non-Interest Expense
```

---

## 📊 PD & LGD Scales

### Probability of Default (PD) Scale

| Rating | PD (%) | Risk Level |
|--------|--------|------------|
| 1 | 0.10% | Minimal |
| 2 | 0.25% | Very Low |
| 3 | 0.50% | Low |
| 4 | 1.00% | Low-Moderate |
| 5 | 2.00% | Moderate |
| 6 | 4.00% | Moderate-High |
| 7 | 8.00% | High |
| 8 | 15.00% | Very High |
| 9 | 25.00% | Severe |
| 10 | 40.00% | Critical |
| 11 | 60.00% | Distressed |
| 12 | 80.00% | Near Default |
| 13 | 95.00% | Default Imminent |

### Loss Given Default (LGD) Scale

| Grade | LGD (%) | Recovery Expectation |
|-------|---------|---------------------|
| A | 10% | Excellent (90% recovery) |
| B | 20% | Very Good (80% recovery) |
| C | 30% | Good (70% recovery) |
| D | 40% | Adequate (60% recovery) |
| E | 50% | Moderate (50% recovery) |
| F | 60% | Poor (40% recovery) |
| G | 75% | Very Poor (25% recovery) |
| H | 90% | Minimal (10% recovery) |

---

## 🛣️ Roadmap

### ✅ Phase 1 (Complete)
- Cash flow calculations
- Present value discounting
- Amortization schedules
- Interactive visualizations
- File upload capability

### 🔄 Phase 2 (In Development)
- **Expected Loss (EL):** PD × LGD × EAD
- **Economic Capital:** Capital required for unexpected losses
- **Regulatory Capital:** Basel III / Reg C calculations
- **RAROC Calculation:** Risk-Adjusted Return / Economic Capital
- **Hurdle Rate Comparison:** RAROC vs. required return
- **Pricing Recommendations:** Suggested rate adjustments

### 🔮 Phase 3 (Planned)
- Stress testing scenarios
- Sensitivity analysis
- Portfolio aggregation
- Batch processing
- API endpoints
- Advanced reporting

---

## 📦 Dependencies

- **dash** (>=4.0.0) - Web application framework
- **plotly** (>=6.5.0) - Interactive visualizations
- **pandas** (>=2.0.0) - Data manipulation
- **numpy** (>=1.26.0) - Numerical computations

See `requirements.txt` for complete list.

---

## 🔧 Technical Details

### Architecture
- **Backend:** Python with Dash framework
- **Frontend:** Dash HTML/Core Components with Plotly charts
- **Calculations:** NumPy and Pandas for financial computations
- **Data Storage:** In-memory DataFrames (CSV export available)

### Performance
- Handles loans up to 360 months (30 years)
- Real-time calculations (< 1 second for typical loans)
- Supports batch processing of multiple loans

---

## 🤝 Contributing

This is a professional banking tool. If you'd like to contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is for educational and professional use.

---

## 📧 Contact

**Author:** Keti Bakiri
**Email:** keti.bakiri.murray@gmail.com

---

## 🏦 Use Cases

This tool is designed for:
- **Retail Banks:** Pricing small business loans
- **Commercial Lenders:** Analyzing term loan profitability
- **Risk Managers:** Understanding risk-adjusted returns
- **Finance Teams:** Portfolio analysis and pricing strategy
- **Students/Educators:** Learning banking and risk management

---

**Built with ❤️ for better banking decisions**

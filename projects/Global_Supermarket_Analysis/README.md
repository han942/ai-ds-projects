# 🛒 Global Supermarket Analysis

> An in-depth analysis of 51,290 global retail transactions using EDA, loss analysis, and multivariate analysis, identifying a structural profit leak: **68% of total losses come from a small number of heavily discounted transactions**.

The results are available both as a static report (PDF) and an **interactive dashboard**.

[Korean README](README_KOR.md)

| | |
|---|---|
| **Data** | Global Supermarket · 51,290 rows × 24 columns · 2019–2022 |
| **Total Sales / Profit** | $12.64M / $1.47M (profit margin **11.6%**) |
| **Key Finding** | Transactions with discounts above 40% represent 13.6% of all transactions but account for **68.2% of total losses** |
| **Stack** | Python (Pandas · NumPy · Matplotlib · Seaborn · SciPy) · MySQL · Vanilla JS dashboard |

---

## 📊 Dashboard

The self-contained HTML dashboard lets you explore the key results in one place. It opens directly in a browser without a server or installation.

- **File:** [`dashboard.html`](dashboard.html) — a self-contained file with the data embedded (double-click to open)
- **Includes:** KPI header · sales/profit trends · regional profit and loss · category profit and loss · discount/loss deep dive · shipping analysis · strategic recommendations
- **Features:** hover tooltips, chart/table toggle, and light/dark theme support

```bash
# Open locally
open dashboard.html        # macOS
xdg-open dashboard.html    # Linux
```

---

## 🗂️ Repository Structure

```
Global_Supermarket_Analysis/
├── README.md                             # This document
├── README_KOR.md                         # Korean documentation
├── global_supermarket.ipynb              # Main analysis notebook (EDA · loss · multivariate · discount)
├── gl_supermarkt_recsys.ipynb            # Data-driven recommendation system experiment
├── global_supermarket_to_sql.ipynb       # Load cleaned data into SQL
├── dashboard.html                        # Interactive dashboard (self-contained)
├── dashboard_build/
│   ├── build_dashboard_data.py           # Generate aggregated JSON from CSV
│   └── dashboard_data.json               # Pre-aggregated data for the dashboard
├── Global_Supermarket_Analysis.pdf       # Final report (English)
├── Global_Supermarket_Analysis_korean.pdf# Final report (Korean)
└── Global_Supermarket_Analysis_db_schema.png  # Normalized schema diagram

../../SQL/global_supermarket/
├── global_supermarket.sql                # DDL/ETL: flat CSV → five normalized tables
└── global_supermarket_ERD.mwb             # MySQL Workbench ERD
```

---

## 🧪 Data & Pipeline

**Raw data** (the Global Supermarket CSV in `datafile/`) contains 51,290 rows and 24 columns with no missing values. It includes customer, order, product, market/region, and shipping information across multiple markets, including US, EU, APAC, LATAM, and Africa.

**① Feature engineering** — custom economic indicators were designed beyond the original metrics:

| Feature | Definition | Purpose |
|---|---|---|
| `pre_sales` | Reconstructed sales before discount | Measure the actual impact of pricing policies |
| `uni_cost` | Unit cost of a product | Evaluate the health of the margin structure |
| `eta` | Elapsed time from order to delivery | Evaluate logistics lead-time performance |

**② Normalization (SQL)** — the flat CSV was decomposed by `row_id` into five tables (`customer` · `product` · `market` · `order` · `shipping`). The schema is available in [`Global_Supermarket_Analysis_db_schema.png`](Global_Supermarket_Analysis_db_schema.png), and the DDL/ETL is documented in [`SQL/global_supermarket/global_supermarket.sql`](../../SQL/global_supermarket/global_supermarket.sql).

**③ Analysis** — EDA across customers, orders, products, markets, and shipping, followed by correlation, loss, profit, multivariate, and discount analyses.

---

## 🔍 Key Findings

### 1. Discounts erode profit

- The Pearson correlation between discount rate and profit is **-0.32** (validated at -0.316), indicating a clear negative relationship.
- Transactions with discounts above 40% account for only **13.6% of all transactions**, but represent **68.2% of total losses**.
  → Losses are concentrated in a small number of heavily discounted transactions rather than spread evenly across the business.

### 2. Tables is the only loss-making subcategory

- **Tables** is the only loss-making subcategory among 17 subcategories, with a net loss of approximately -$64K.
- **Bevis, Barricks, and Lesro** are the three manufacturers with the largest losses.

### 3. Regional profit and loss variation

- Total profit ranks **APAC > EU > US > LATAM**, while larger markets also tend to carry larger losses.
- In APAC, US, and LATAM, table sales contribute disproportionately to each market's overall losses.

### 4. Growing transaction volume with clear seasonality

- Transaction volume grew from 8,998 in 2019 to 17,531 in 2022, an increase of approximately **95%** (**+27.0% year over year in 2022**).
- A clear seasonal peak recurs every year in the second half, especially from September through December.

---

## 💡 Strategic Recommendations

| # | Recommendation | Rationale |
|---|---|---|
| 1 | **Introduce a 40% discount cap for high-volume categories** (Phones, Bookcases, etc.) | Discounts above 40% cause 68% of total losses |
| 2 | **Reduce the average discount rate in the EU by 20 percentage points** | Correct the EU's large loss burden and regional profitability |
| 3 | **Reduce or discontinue table products from high-loss manufacturers** (Bevis, Barricks, Lesro) | Tables is the only loss-making subcategory and these manufacturers lead losses |
| 4 | **Redesign pricing for high-loss cities** (Istanbul, Lagos, etc.) | Reconfigure discounts within the 40% threshold |

---

## 🔁 Reproduction

```bash
# 1. Regenerate aggregated data for the dashboard (requires the raw CSV)
python dashboard_build/build_dashboard_data.py
#    → updates dashboard_build/dashboard_data.json
#    (prints total sales/profit, correlation -0.316, and the 68.2% high-discount loss share)

# 2. Open the dashboard
open dashboard.html

# 3. Reproduce the full analysis
jupyter notebook global_supermarket.ipynb
```

> Note: `dashboard.html` is a static file with the aggregated results embedded, so it can be opened without the CSV. Run step 1 again after updating the raw data to refresh the dashboard.

---

## 🧰 Tech Stack

- **Languages:** Python, SQL, JavaScript
- **Analysis:** Pandas · NumPy · SciPy (statistics) · Matplotlib · Seaborn (visualization)
- **Database:** MySQL (normalized schema design · ETL)
- **Dashboard:** Dependency-free Vanilla JS + SVG (no external libraries)
- **Methods:** Multivariate EDA · feature engineering · correlation analysis · risk profiling · BI reporting

"""Aggregate the Global Supermarket CSV into a compact JSON for dashboard.html.

Reproduces the groupby logic from global_supermarket.ipynb (loss/profit split
by sub_category & market_area, discount-bucket loss share, Tables manufacturer
breakdown, monthly trend, ship/eta distribution, discount-profit correlation)
without pandas, since this environment only has the stdlib.
"""
import csv
import json
import statistics
from collections import defaultdict
from datetime import date

CSV_PATH = "/home/hananthony1/codes/ai-ds-projects/datafile/[DArt-B 5기] Global_Supermarket.csv"
OUT_PATH = "/home/hananthony1/codes/ai-ds-projects/projects/Global_Supermarket_Analysis/dashboard_data.json"


def parse_date(s):
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


rows = []
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "sales": float(r["sales"]),
            "profit": float(r["profit"]),
            "discount": float(r["discount"]),
            "quantity": float(r["quantity"]),
            "order_date": parse_date(r["order_date"]),
            "ship_date": parse_date(r["ship_date"]),
            "order_year": int(r["order_year"]),
            "category": r["category"],
            "sub_category": r["sub_category"],
            "market_area": r["market_area"],
            "order_region": r["oreder_region"],
            "ship_mode": r["ship_mode"],
            "product_name": r["product_name"],
        })

n = len(rows)

# ---- top-line KPIs ----
total_sales = sum(r["sales"] for r in rows)
total_profit = sum(r["profit"] for r in rows)
avg_discount = sum(r["discount"] for r in rows) / n

by_year_count = defaultdict(int)
for r in rows:
    by_year_count[r["order_year"]] += 1
years_sorted = sorted(by_year_count)
yoy = {}
for i in range(1, len(years_sorted)):
    y0, y1 = years_sorted[i - 1], years_sorted[i]
    yoy[str(y1)] = round((by_year_count[y1] - by_year_count[y0]) / by_year_count[y0] * 100, 1)

# ---- monthly trend ----
monthly = defaultdict(lambda: {"sales": 0.0, "profit": 0.0, "count": 0})
for r in rows:
    key = f'{r["order_date"].year:04d}-{r["order_date"].month:02d}'
    m = monthly[key]
    m["sales"] += r["sales"]
    m["profit"] += r["profit"]
    m["count"] += 1
monthly_series = [
    {"month": k, "sales": round(v["sales"], 2), "profit": round(v["profit"], 2), "count": v["count"]}
    for k, v in sorted(monthly.items())
]

# ---- gain/loss split helper ----
def split_group(key_fn):
    total = defaultdict(float)
    loss = defaultdict(float)
    gain = defaultdict(float)
    for r in rows:
        k = key_fn(r)
        p = r["profit"]
        total[k] += p
        if p < 0:
            loss[k] += p
        else:
            gain[k] += p
    keys = set(total) | set(loss) | set(gain)
    return [
        {"key": k, "total": round(total[k], 2), "loss": round(loss[k], 2), "gain": round(gain[k], 2)}
        for k in keys
    ]

by_sub_category = sorted(split_group(lambda r: r["sub_category"]), key=lambda x: x["total"])
by_market_area = sorted(split_group(lambda r: r["market_area"]), key=lambda x: x["total"])
by_order_region = sorted(split_group(lambda r: r["order_region"]), key=lambda x: x["total"])

# ---- discount buckets ----
bucket_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
def bucket_of(discount_pct):
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        if lo <= discount_pct < hi or (hi == 100 and discount_pct == 100):
            return f"{lo}-{hi}%"
    return f"{bucket_edges[-1]}%+"

bucket_count = defaultdict(int)
bucket_loss = defaultdict(float)
bucket_gain = defaultdict(float)
for r in rows:
    b = bucket_of(r["discount"] * 100)
    bucket_count[b] += 1
    if r["profit"] < 0:
        bucket_loss[b] += r["profit"]
    else:
        bucket_gain[b] += r["profit"]

bucket_order = [f"{bucket_edges[i]}-{bucket_edges[i+1]}%" for i in range(len(bucket_edges) - 1)]
discount_buckets = [
    {
        "bucket": b,
        "count": bucket_count[b],
        "loss": round(bucket_loss[b], 2),
        "gain": round(bucket_gain[b], 2),
    }
    for b in bucket_order
]

# ---- 40%+ discount headline stats (mirrors notebook cell 79) ----
hi_disc = [r for r in rows if r["discount"] > 0.4]
all_gain_sum = sum(r["profit"] for r in rows if r["profit"] > 0)
all_loss_sum = sum(r["profit"] for r in rows if r["profit"] < 0)
hi_disc_gain_sum = sum(r["profit"] for r in hi_disc if r["profit"] > 0)
hi_disc_loss_sum = sum(r["profit"] for r in hi_disc if r["profit"] < 0)

headline = {
    "hi_discount_txn_share_pct": round(len(hi_disc) / n * 100, 1),
    "hi_discount_gain_share_pct": round(hi_disc_gain_sum / all_gain_sum * 100, 1) if all_gain_sum else None,
    "hi_discount_loss_share_pct": round(hi_disc_loss_sum / all_loss_sum * 100, 1) if all_loss_sum else None,
}

# ---- Tables sub-category manufacturer breakdown ----
tables = [r for r in rows if r["sub_category"] == "Tables"]
mfr_profit = defaultdict(float)
for r in tables:
    name_short = r["product_name"].split(" ")[0]
    mfr_profit[name_short] += r["profit"]
tables_by_manufacturer = sorted(
    [{"manufacturer": k, "profit": round(v, 2)} for k, v in mfr_profit.items()],
    key=lambda x: x["profit"],
)

# ---- shipping ----
ship_mode_count = defaultdict(int)
for r in rows:
    ship_mode_count[r["ship_mode"]] += 1
ship_mode_dist = sorted(
    [{"mode": k, "count": v} for k, v in ship_mode_count.items()], key=lambda x: -x["count"]
)

eta_count = defaultdict(int)
for r in rows:
    eta = (r["ship_date"] - r["order_date"]).days
    eta_count[eta] += 1
eta_dist = sorted([{"eta": k, "count": v} for k, v in eta_count.items()], key=lambda x: x["eta"])

# ---- discount vs profit correlation (Pearson) ----
discounts = [r["discount"] for r in rows]
profits = [r["profit"] for r in rows]
correlation = statistics.correlation(discounts, profits)

output = {
    "meta": {"n_transactions": n},
    "kpi": {
        "total_sales": round(total_sales, 2),
        "total_profit": round(total_profit, 2),
        "margin_pct": round(total_profit / total_sales * 100, 2),
        "avg_discount_pct": round(avg_discount * 100, 2),
        "n_transactions": n,
        "yoy_transaction_growth_pct": yoy,
        "discount_profit_correlation": round(correlation, 3),
    },
    "monthly_series": monthly_series,
    "by_sub_category": by_sub_category,
    "by_market_area": by_market_area,
    "by_order_region": by_order_region,
    "discount_buckets": discount_buckets,
    "headline": headline,
    "tables_by_manufacturer": tables_by_manufacturer,
    "ship_mode_dist": ship_mode_dist,
    "eta_dist": eta_dist,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Wrote", OUT_PATH)
print("n_transactions:", n)
print("total_sales:", round(total_sales, 2), "total_profit:", round(total_profit, 2))
print("margin_pct:", round(total_profit / total_sales * 100, 2))
print("avg_discount_pct:", round(avg_discount * 100, 2))
print("yoy_transaction_growth_pct:", yoy)
print("discount_profit_correlation:", round(correlation, 3))
print("headline (40%+ discount):", headline)

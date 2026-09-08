"""Source reconciliation checks: python3 -m unittest discover -s dashboard_build."""
import json
import unittest
from collections import Counter
from pathlib import Path

from build_dashboard_data import CSV_PATH, PROJECT_DIR, read_rows


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_rows(CSV_PATH)
        cls.data = json.loads((Path(__file__).parent / 'dashboard_data.json').read_text())

    def test_html_embeds_the_generated_snapshot(self):
        html = (PROJECT_DIR / 'dashboard.html').read_text()
        payload = html.split('const DATA = ', 1)[1].split(';\n', 1)[0]
        self.assertEqual(json.loads(payload), self.data)

    def test_aggregates_reconcile_for_all_periods(self):
        for period, data in [('all', self.data), *self.data['by_year'].items()]:
            with self.subTest(period=period):
                rows = [r for r in self.rows if period == 'all' or str(r['order_year']) == period]
                kpi = data['kpi']
                self.assertEqual(len(rows), kpi['n_transactions'])
                self.assertAlmostEqual(sum(r['sales'] for r in rows), kpi['total_sales'], places=2)
                self.assertAlmostEqual(sum(r['profit'] for r in rows), kpi['total_profit'], places=2)
                self.assertAlmostEqual(kpi['total_profit'] / kpi['total_sales'] * 100, kpi['margin_pct'], places=2)
                for group in ('monthly_series', 'discount_buckets', 'ship_mode_dist', 'eta_dist'):
                    self.assertEqual(sum(r['count'] for r in data[group]), len(rows), group)
                for group in ('by_market_area', 'by_order_region', 'by_sub_category'):
                    self.assertAlmostEqual(sum(r['total'] for r in data[group]), kpi['total_profit'], delta=0.1)
                    for r in data[group]:
                        self.assertAlmostEqual(r['gain'] + r['loss'], r['total'], delta=0.011)
                loss = sum(r['profit'] for r in rows if r['profit'] < 0)
                high = [r for r in rows if r['discount'] > 0.4]
                self.assertEqual(round(len(high) / len(rows) * 100, 1), data['headline']['hi_discount_txn_share_pct'])
                self.assertEqual(round(sum(r['profit'] for r in high if r['profit'] < 0) / loss * 100, 1), data['headline']['hi_discount_loss_share_pct'])
                self.assertEqual(Counter((r['ship_date'] - r['order_date']).days for r in rows), {r['eta']: r['count'] for r in data['eta_dist']})

    def test_strict_discount_threshold(self):
        self.assertEqual(sum(r['discount'] > 0.4 for r in self.rows), 6961)
        self.assertEqual(sum(r['discount'] >= 0.4 for r in self.rows), 10138)


if __name__ == '__main__':
    unittest.main()

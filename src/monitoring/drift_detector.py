"""Evidently AI Data & Feature Drift Detector for Sentinel AI."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import settings
from src.utils.logger import logger
from src.monitoring.prometheus_exporter import DATA_DRIFT_SCORE_GAUGE


class DriftDetector:
    """Detects statistical covariate shift and sensor distribution drift between baseline and production."""

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or settings.REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.reference_data: Optional[pd.DataFrame] = None

    def load_reference_data(self) -> pd.DataFrame:
        """Loads baseline reference dataset (training distribution)."""
        ref_path = settings.DATA_PROCESSED_DIR / "train_FD001.parquet"
        if ref_path.exists():
            df = pd.read_parquet(ref_path)
        else:
            from src.ingestion.cmapss_loader import CMAPSSDataLoader
            loader = CMAPSSDataLoader()
            df, _, _ = loader.load_or_generate_data("FD001")
        self.reference_data = df
        return df

    def run_drift_analysis(
        self, current_data: pd.DataFrame, drift_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Runs comparative drift detection between baseline and incoming production telemetry."""
        if self.reference_data is None:
            self.load_reference_data()

        eval_cols = settings.INFORMATIVE_SENSOR_COLS
        reference_sub = self.reference_data[eval_cols].dropna()
        current_sub = current_data[eval_cols].dropna()

        if len(current_sub) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 production samples for statistical testing; got {len(current_sub)}.",
                "drift_detected": False,
                "drift_share": 0.0,
                "drifted_features": [],
            }

        drift_results = {}
        drifted_features = []

        # 1. Statistical Two-Sample Kolmogorov-Smirnov Test per sensor channel
        for col in eval_cols:
            ref_series = reference_sub[col].values
            curr_series = current_sub[col].values

            # KS-test evaluates hypothesis that both samples are drawn from same distribution
            ks_stat, p_value = ks_2samp(ref_series, curr_series)
            is_drifted = bool(p_value < drift_threshold)
            
            ref_mean = float(np.mean(ref_series))
            curr_mean = float(np.mean(curr_series))
            mean_shift_pct = float(round(((curr_mean - ref_mean) / (abs(ref_mean) + 1e-6)) * 100, 2))

            drift_results[col] = {
                "p_value": round(float(p_value), 5),
                "ks_statistic": round(float(ks_stat), 4),
                "is_drifted": is_drifted,
                "baseline_mean": round(ref_mean, 2),
                "current_mean": round(curr_mean, 2),
                "mean_shift_pct": mean_shift_pct,
            }
            if is_drifted:
                drifted_features.append(col)

        drift_share = round(len(drifted_features) / len(eval_cols), 3)
        overall_drift_detected = drift_share >= 0.30  # If 30%+ of sensors drift, flag dataset drift

        # Update Prometheus gauge
        DATA_DRIFT_SCORE_GAUGE.set(drift_share * 100)

        # 2. Try generating full Evidently visual report if available
        html_report_path = self.reports_dir / "drift_report.html"
        json_report_path = self.reports_dir / "drift_summary.json"

        evidently_success = False
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_sub, current_data=current_sub)
            report.save_html(str(html_report_path))
            evidently_success = True
            logger.info(f"Evidently HTML report generated at {html_report_path}")
        except Exception as e:
            logger.debug(f"Evidently report render notice: {e}. Using pure KS-test report.")
            # Create a clean fallback HTML report
            self._write_html_report(html_report_path, drift_results, drift_share, overall_drift_detected)

        summary = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "drift_detected": overall_drift_detected,
            "drift_share": drift_share,
            "drifted_features_count": len(drifted_features),
            "total_features_evaluated": len(eval_cols),
            "drifted_features": drifted_features,
            "feature_details": drift_results,
            "html_report_path": str(html_report_path),
        }

        with open(json_report_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Drift Analysis Complete: Drift Detected={overall_drift_detected}, "
            f"Drift Share={drift_share*100:.1f}%, Drifted Channels={drifted_features}"
        )
        return summary

    def _write_html_report(
        self,
        output_path: Path,
        drift_results: Dict[str, Any],
        drift_share: float,
        overall_drift: bool,
    ):
        """Generates a standalone styled HTML report."""
        status_color = "#e53e3e" if overall_drift else "#38a169"
        status_text = "DATA DRIFT DETECTED" if overall_drift else "DISTRIBUTION NORMAL"

        rows_html = ""
        for feat, data in drift_results.items():
            badge = "<span style='color: #e53e3e; font-weight: bold;'>DRIFT</span>" if data["is_drifted"] else "<span style='color: #38a169;'>STABLE</span>"
            desc = settings.SENSOR_DESCRIPTIONS.get(feat, feat)
            rows_html += f"""
            <tr>
                <td><strong>{feat}</strong> ({desc})</td>
                <td>{data['baseline_mean']}</td>
                <td>{data['current_mean']}</td>
                <td>{data['mean_shift_pct']}%</td>
                <td>{data['p_value']}</td>
                <td>{badge}</td>
            </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentinel AI - Sensor Drift Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }}
                h1 {{ margin-bottom: 5px; }}
                .banner {{ background: {status_color}; padding: 15px 25px; border-radius: 8px; font-size: 20px; font-weight: bold; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
                th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
                th {{ background: #0f172a; color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
                tr:hover {{ background: #283548; }}
            </style>
        </head>
        <body>
            <h1>Sentinel AI — Automated Telemetry Drift Report</h1>
            <p style="color: #94a3b8;">Generated by Evidently AI / SciPy Statistical Evaluation Engine</p>
            <div class="banner">{status_text} (Drift Share: {round(drift_share*100, 1)}%)</div>
            <table>
                <thead>
                    <tr>
                        <th>Sensor Channel</th>
                        <th>Baseline Mean</th>
                        <th>Current Mean</th>
                        <th>Mean Shift</th>
                        <th>P-Value (KS-Test)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </body>
        </html>
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


drift_detector = DriftDetector()

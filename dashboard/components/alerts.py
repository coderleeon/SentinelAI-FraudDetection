"""
Alert Management
================
Handles fraud alerts across all notification channels:
  • Structured console logging (always on)
  • Append-only alert log file
  • Optional email (SMTP) via environment variables
  • Extensible: add Slack / Telegram / PagerDuty here

Environment variables for email alerts:
  ALERT_EMAIL           — sender address
  ALERT_EMAIL_TO        — recipient (defaults to sender)
  ALERT_EMAIL_PASSWORD  — SMTP password / app-token
  SMTP_SERVER           — defaults to smtp.gmail.com
  SMTP_PORT             — defaults to 587
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Central alert dispatcher.  Thread-safe for use in async FastAPI handlers
    and synchronous Streamlit callbacks.
    """

    def __init__(self):
        self.alert_count   = 0
        self._email_cfg    = self._load_email_config()
        self._alert_logger = self._setup_file_logger()

    # ── Public API ────────────────────────────────────────────────────────────

    def emit(
        self,
        transaction_id: str,
        amount:         float,
        risk_score:     float,
        risk_level:     str,
        fraud_prob:     float,
        reasons:        List[str],
    ) -> str:
        """
        Fire all configured alert channels and return the formatted message string.
        Suitable for both sync and async callers (no await needed).
        """
        self.alert_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trigger_str = "; ".join(reasons) if reasons else "ML model detection"

        msg = (
            f"🚨 FRAUD #{self.alert_count:05d} | "
            f"{timestamp} | "
            f"TXN={transaction_id} | "
            f"${amount:,.2f} | "
            f"Risk={risk_score:.4f} ({risk_level}) | "
            f"Prob={fraud_prob:.4f} | "
            f"Triggers: {trigger_str}"
        )

        # 1. Console
        logger.warning(msg)

        # 2. File
        self._alert_logger.warning(msg)

        # 3. Email (fire-and-forget — failures are non-fatal)
        if self._email_cfg:
            try:
                self._send_email(transaction_id, amount, risk_score, risk_level, fraud_prob, reasons)
            except Exception as exc:
                logger.debug(f"Email alert failed (non-critical): {exc}")

        return msg

    def get_recent_alerts(self, n: int = 50) -> List[str]:
        """Read the last n lines from the alert log file."""
        log_path = self._alert_logger.handlers[0].baseFilename if self._alert_logger.handlers else None
        if not log_path or not os.path.exists(log_path):
            return []
        with open(log_path) as f:
            lines = f.readlines()
        return [l.strip() for l in lines[-n:]]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _setup_file_logger(self) -> logging.Logger:
        _ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_path = os.path.join(_ROOT, "models", "fraud_alerts.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        al = logging.getLogger("fraud_alerts")
        if not al.handlers:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            al.addHandler(fh)
        al.setLevel(logging.WARNING)
        al.propagate = False
        return al

    def _load_email_config(self) -> Optional[dict]:
        sender = os.getenv("ALERT_EMAIL", "")
        if not sender:
            return None
        return {
            "sender":   sender,
            "to":       os.getenv("ALERT_EMAIL_TO", sender),
            "password": os.getenv("ALERT_EMAIL_PASSWORD", ""),
            "server":   os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "port":     int(os.getenv("SMTP_PORT", "587")),
        }

    def _send_email(
        self,
        txn_id:     str,
        amount:     float,
        risk_score: float,
        risk_level: str,
        fraud_prob: float,
        reasons:    List[str],
    ):
        cfg = self._email_cfg
        if not cfg or not cfg["password"]:
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 Fraud Alert — {txn_id}"
        msg["From"]    = cfg["sender"]
        msg["To"]      = cfg["to"]

        html_body = f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#0F172A;color:#E2E8F0;padding:24px">
          <h2 style="color:#EF4444">🚨 Fraudulent Transaction Detected</h2>
          <table style="border-collapse:collapse;width:100%;max-width:600px">
            <tr><td style="padding:8px;border:1px solid #334155"><b>Transaction ID</b></td>
                <td style="padding:8px;border:1px solid #334155">{txn_id}</td></tr>
            <tr><td style="padding:8px;border:1px solid #334155"><b>Amount</b></td>
                <td style="padding:8px;border:1px solid #334155">${amount:,.2f}</td></tr>
            <tr><td style="padding:8px;border:1px solid #334155"><b>Risk Score</b></td>
                <td style="padding:8px;border:1px solid #334155">{risk_score:.4f}</td></tr>
            <tr><td style="padding:8px;border:1px solid #334155"><b>Risk Level</b></td>
                <td style="padding:8px;border:1px solid #334155">{risk_level}</td></tr>
            <tr><td style="padding:8px;border:1px solid #334155"><b>Fraud Probability</b></td>
                <td style="padding:8px;border:1px solid #334155">{fraud_prob:.4f}</td></tr>
            <tr><td style="padding:8px;border:1px solid #334155"><b>Rule Triggers</b></td>
                <td style="padding:8px;border:1px solid #334155">{"<br>".join(reasons) or "ML detection"}</td></tr>
          </table>
          <p style="color:#94A3B8;font-size:12px;margin-top:24px">
            Sent by AI Fraud Detection System v2.0
          </p>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(cfg["server"], cfg["port"], timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["sender"], cfg["password"])
            server.sendmail(cfg["sender"], cfg["to"], msg.as_string())
        logger.info(f"Email alert sent to {cfg['to']}")


# ── Module-level singleton ────────────────────────────────────────────────────
alert_manager = AlertManager()

"""Launch the Streamlit dashboard."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    port = os.getenv("DASHBOARD_PORT", "8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join("dashboard", "app.py"),
        "--server.port", port,
        "--server.headless", "false",
    ])

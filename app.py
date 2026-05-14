"""Streamlit entry point. Run with: streamlit run app.py"""

from src.utils.logger import setup_logger
from src.ui.dashboard import main

setup_logger("dashboard")

if __name__ == "__main__":
    main()

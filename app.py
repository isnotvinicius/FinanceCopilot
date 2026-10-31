"""Ponto de entrada: streamlit run app.py"""

import logging

from finance_copilot.ui.app import main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

main()

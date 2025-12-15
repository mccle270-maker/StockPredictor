#!/usr/bin/env bash
python3 -m streamlit run app.py --server.port 8509 &
ngrok http 8509
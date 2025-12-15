import time
import schedule
from auto_paper_trade import main as trade_once

trade_once()  # run immediately once [web:344]

schedule.every().monday.at("08:35").do(trade_once)

while True:
    schedule.run_pending()
    time.sleep(1)
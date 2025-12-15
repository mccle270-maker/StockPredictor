import time
import schedule

from auto_paper_trade import main as trade_once

schedule.every().monday.at("08:35").do(trade_once)   # adjust for market open + your timezone
# or: schedule.every(5).minutes.do(trade_once)

while True:
    schedule.run_pending()
    time.sleep(1)
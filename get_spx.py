import yfinance as yf

spx = yf.Ticker("^GSPC")
hist = spx.history(period="5d")

print(hist)


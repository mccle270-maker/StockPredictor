"""
SEC EDGAR provider - Free fundamentals from official filings.
No API key required!
"""
from __future__ import annotations

import pandas as pd
import requests
from typing import Optional, Dict, Any
from datetime import datetime
import time

from .base import BaseProvider, ProviderResponse


class SECEdgarProvider(BaseProvider):
    """
    SEC EDGAR provider for official fundamentals from 10-K/10-Q filings.
    
    Pros:
    - Free, no API key needed
    - Official company data
    - Reliable, authoritative source
    
    Cons:
    - Only US companies
    - Data may be delayed (quarterly updates)
    - Complex parsing required
    """
    
    name = "sec_edgar"
    requires_key = False
    rate_limit_per_minute = 10  # SEC asks for max 10 req/sec but we go slow
    
    BASE_URL = "https://data.sec.gov"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    
    # Required headers per SEC guidelines
    HEADERS = {
        "User-Agent": "StockPredictor/1.0 (contact@example.com)",
        "Accept-Encoding": "gzip, deflate",
    }
    
    def __init__(self):
        super().__init__()
        self._cik_cache: Dict[str, str] = {}
    
    def is_available(self) -> bool:
        return True  # Always available, no key needed
    
    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        
        try:
            resp = requests.get(self.COMPANY_TICKERS_URL, headers=self.HEADERS, timeout=10)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            # Build ticker -> CIK mapping
            for entry in data.values():
                tk = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                self._cik_cache[tk] = cik
            
            return self._cik_cache.get(ticker.upper())
            
        except Exception:
            return None
    
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> ProviderResponse:
        """SEC EDGAR does not provide price history."""
        return ProviderResponse(
            success=False,
            data=None,
            source=self.name,
            error="SEC EDGAR does not provide price history",
        )
    
    def get_fundamentals(self, ticker: str) -> ProviderResponse:
        """Fetch fundamentals from SEC EDGAR company facts."""
        try:
            self._log_request()
            
            cik = self._get_cik(ticker)
            if not cik:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"CIK not found for {ticker}",
                )
            
            # Rate limit compliance
            time.sleep(0.2)
            
            # Get company facts
            url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code} for CIK {cik}",
                )
            
            company_data = resp.json()
            
            # Get company facts (contains financials)
            time.sleep(0.2)
            facts_url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
            facts_resp = requests.get(facts_url, headers=self.HEADERS, timeout=15)
            
            fundamentals = {
                "company_name": company_data.get("name"),
                "sic": company_data.get("sic"),
                "sic_description": company_data.get("sicDescription"),
                "fiscal_year_end": company_data.get("fiscalYearEnd"),
            }
            
            if facts_resp.status_code == 200:
                facts = facts_resp.json()
                
                # Extract key metrics from US-GAAP facts
                us_gaap = facts.get("facts", {}).get("us-gaap", {})
                
                # Get most recent values for key metrics
                fundamentals.update(self._extract_metrics(us_gaap))
            
            has_data = fundamentals.get("company_name") is not None
            
            return ProviderResponse(
                success=has_data,
                data=fundamentals,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
    
    def _extract_metrics(self, us_gaap: Dict) -> Dict[str, Any]:
        """Extract key financial metrics from US-GAAP facts."""
        metrics = {}
        
        # Key metrics to extract
        metric_mappings = {
            "NetIncomeLoss": "fund_net_income",
            "Revenues": "fund_revenue",
            "Assets": "fund_total_assets",
            "Liabilities": "fund_total_liabilities",
            "StockholdersEquity": "fund_stockholders_equity",
            "EarningsPerShareBasic": "fund_eps_basic",
            "EarningsPerShareDiluted": "fund_eps_diluted",
            "CommonStockSharesOutstanding": "fund_shares_outstanding",
            "CashAndCashEquivalentsAtCarryingValue": "fund_cash",
            "LongTermDebt": "fund_long_term_debt",
            "OperatingIncomeLoss": "fund_operating_income",
            "GrossProfit": "fund_gross_profit",
        }
        
        for xbrl_name, metric_name in metric_mappings.items():
            if xbrl_name in us_gaap:
                try:
                    units = us_gaap[xbrl_name].get("units", {})
                    # Usually in USD or shares
                    for unit_type, values in units.items():
                        if values:
                            # Get most recent 10-K value
                            for val in reversed(values):
                                if val.get("form") == "10-K":
                                    metrics[metric_name] = val.get("val")
                                    break
                            else:
                                # Fallback to most recent value
                                metrics[metric_name] = values[-1].get("val")
                            break
                except Exception:
                    pass
        
        # Calculate P/E if we have the data
        if metrics.get("fund_eps_diluted") and metrics.get("fund_eps_diluted") > 0:
            # We'd need current price, but we don't have it here
            # Leave for aggregator to calculate
            pass
        
        # Calculate P/B if we have the data
        if metrics.get("fund_stockholders_equity") and metrics.get("fund_shares_outstanding"):
            book_value_per_share = metrics["fund_stockholders_equity"] / metrics["fund_shares_outstanding"]
            metrics["fund_book_value_per_share"] = book_value_per_share
        
        return metrics
    
    def get_filings(self, ticker: str, form_type: str = "10-K", limit: int = 5) -> ProviderResponse:
        """Get recent SEC filings for a ticker."""
        try:
            self._log_request()
            
            cik = self._get_cik(ticker)
            if not cik:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"CIK not found for {ticker}",
                )
            
            time.sleep(0.2)
            
            url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code}",
                )
            
            data = resp.json()
            filings = data.get("filings", {}).get("recent", {})
            
            # Extract filings of requested type
            results = []
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            
            for i, form in enumerate(forms):
                if form == form_type and len(results) < limit:
                    results.append({
                        "form": form,
                        "filing_date": dates[i] if i < len(dates) else None,
                        "accession_number": accessions[i] if i < len(accessions) else None,
                    })
            
            return ProviderResponse(
                success=True,
                data=results,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )

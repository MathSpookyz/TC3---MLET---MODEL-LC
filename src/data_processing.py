import pandas as pd
import yfinance as yf

def fetch_financial_metrics(tickers):
    """
    Busca métricas financeiras atuais para múltiplos tickers usando yfinance.
    Retorna DataFrame pronto para pré-processamento.
    """
    data_list = []

    for t in tickers:
        ticker = yf.Ticker(t)
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet

        try:
            net_income = financials.loc['Net Income'].iloc[0]
        except Exception:
            net_income = None

        try:
            total_equity = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]
        except Exception:
            total_equity = None

        try:
            total_debt = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
        except Exception:
            total_debt = None

        data_list.append({
            'ticker': t,
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins'),
            'beta': info.get('beta'),
            'market_cap': info.get('marketCap'),
            'net_income': net_income,
            'total_equity': total_equity,
            'total_debt': total_debt
        })

    df = pd.DataFrame(data_list)
    return df
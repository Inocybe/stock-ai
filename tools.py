import inspect
import logging
import yfinance as yf


logger = logging.getLogger(__name__)



def get_stock_price(ticker: str) -> str:
    """
    Fetches the current stock price for the given ticker symbol using yfinance.
    """
    try:
        return str(yf.Ticker(ticker).fast_info['last_price'])
    except Exception as e:
        logger.error(f"Error fetching stock price for {ticker}: {e}")
        return "Error fetching stock price"


def get_stock_info(ticker: str) -> str:
    """
    Fetches the stock info for the given ticker symbol using yfinance.
    """
    try:
        return str(yf.Ticker(ticker).info)
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        return "Error fetching stock info"

def get_news(ticker: str) -> str:
    """
    Fetches the news for the given ticker symbol using yfinance.
    """
    try:
        return str(yf.Ticker(ticker).news)
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return "Error fetching news"



def get_stock_history(ticker: str, period='1d', interval='1m') -> str:
    """
    Fetches the stock history for the given ticker symbol using yfinance.
    """
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return data.to_string()
    except Exception as e:
        logger.error(f"Error fetching stock history for {ticker}: {e}")
        return "Error fetching stock history"


def get_financials(ticker: str) -> str:
    """
    Fetches the financials for the given ticker symbol using yfinance.
    """
    try:
        return str(yf.Ticker(ticker).financials)
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")
        return "Error fetching financials"

def get_balance_sheet(ticker: str) -> str:
    """
    Fetches the balance sheet for the given ticker symbol using yfinance.
    """
    try:
        return str(yf.Ticker(ticker).balance_sheet)
    except Exception as e:
        logger.error(f"Error fetching balance sheet for {ticker}: {e}")
        return "Error fetching balance sheet"

def get_cashflow(ticker: str) -> str:
    """
    Fetches the cash flow for the given ticker symbol using yfinance.
    """
    try:
        return str(yf.Ticker(ticker).cashflow)
    except Exception as e:
        logger.error(f"Error fetching cashflow for {ticker}: {e}")
        return "Error fetching cashflow"

'''
def get_sec_filings(ticker: str, filing_type='10-K', count=5) -> str:
    try:
        return dl.get(ticker, filing_type, count)
    except Exception as e:
        logger.error(f"Error fetching SEC filings for {ticker}: {e}")
        return "Error fetching SEC filings"
'''





tool_list: dict[str, callable] = {
    "get_stock_price": get_stock_price,
    "get_stock_history": get_stock_history,
    "get_news": get_news,
    "get_stock_info": get_stock_info,
    "get_financials": get_financials,
    "get_balance_sheet": get_balance_sheet,
    "get_cashflow": get_cashflow
}



def py_to_json(py_type: type) -> dict:
    """
    Converts a Python type to a JSON schema type.
    """
    type_map = {
        str:   {"type": "string"},
        int:   {"type": "integer"},
        float: {"type": "number"},
        bool:  {"type": "boolean"},
        list:  {"type": "array", "items": {}},
        dict:  {"type": "object"},
    }
    if py_type not in type_map:
        raise ValueError(f"Unsupported type: {py_type}")
    return type_map[py_type]


# use standard OpenAI format
def define_tool(func: callable) -> dict:
    sig = inspect.signature(func)
    annotations = {k: v for k, v in func.__annotations__.items() if k != 'return'}


    required = []
    for name, param in sig.parameters.items():
        if name not in annotations:
            continue
        # no default = required
        if param.default is inspect.Parameter.empty:
            required.append(name)


    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {
                    name: py_to_json(typ) for name, typ in annotations.items()
                }
            },
            "required": required
        }
    }



def define_all_tools() -> list:
    tls = []
    for func in tool_list.values():
        tls.append(define_tool(func))
    return tls




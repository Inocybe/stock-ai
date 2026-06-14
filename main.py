
from ollama import ChatResponse, Client
from sec_edgar_downloader import Downloader
import logging
import yfinance as yf


# ---- SETUP LOGGING ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- DOWNLOADING SHIT ----
dl = Downloader("Lev Mitchell", "levmitchell@icloud.com")




def get_stock_price(ticker) -> str:
    try:
        return str(yf.Ticker(ticker).fast_info['last_price'])
    except Exception as e:
        logger.error(f"Error fetching stock price for {ticker}: {e}")
        return "Error fetching stock price"


def get_stock_info(ticker) -> str:
    try:
        return str(yf.Ticker(ticker).info)
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        return "Error fetching stock info"

def get_news(ticker) -> str:
    try:
        return str(yf.Ticker(ticker).news)
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return "Error fetching news"


def get_stock_history(ticker, period='1d', interval='1m') -> str:
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return data.to_string()
    except Exception as e:
        logger.error(f"Error fetching stock history for {ticker}: {e}")
        return "Error fetching stock history"


def get_financials(ticker) -> str:
    try:
        return str(yf.Ticker(ticker).financials)
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")
        return "Error fetching financials"

def get_balance_sheet(ticker) -> str:
    try:
        return str(yf.Ticker(ticker).balance_sheet)
    except Exception as e:
        logger.error(f"Error fetching balance sheet for {ticker}: {e}")
        return "Error fetching balance sheet"

def get_cashflow(ticker) -> str:
    try:
        return str(yf.Ticker(ticker).cashflow)
    except Exception as e:
        logger.error(f"Error fetching cashflow for {ticker}: {e}")
        return "Error fetching cashflow"

'''
def get_sec_filings(ticker, filing_type='10-K', count=5) -> str:
    try:
        return dl.get(ticker, filing_type, count)
    except Exception as e:
        logger.error(f"Error fetching SEC filings for {ticker}: {e}")
        return "Error fetching SEC filings"
'''

options = {
    'temperature': 0.7,
    'num_ctx': 16384,
    'top_p': 0.9
}

tools = [
    get_stock_price,
    get_stock_history,
    #get_sec_filings,
    get_news,
    get_stock_info,
    get_financials,
    get_balance_sheet,
    get_cashflow
]

available_functions = {
    'get_stock_price': get_stock_price,
    'get_stock_history': get_stock_history,
    # 'get_sec_filings': get_sec_filings,
    'get_news': get_news,
    'get_stock_info': get_stock_info,
    'get_financials': get_financials,
    'get_balance_sheet': get_balance_sheet,
    'get_cashflow': get_cashflow
}

def main():
    client = Client()
    model = "qwen3.5:9b"

    user_input = input("Enter your question: ")

    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides stock price information."},
        {"role": "user", "content": user_input}
    ]

    # first chat call
    response = client.chat(model=model, messages=messages, tools=tools, think=True, options=options)
    
    # Look through for as many times as model wants to call tools
    while response.message.tool_calls:
        messages.append(response.message)

        for tool_call in response.message.tool_calls:
            if function_to_call := available_functions.get(tool_call.function.name):
                logger.info(f"Calling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
                result = function_to_call(**tool_call.function.arguments)
                #logger.info(f"Tool result: {result}")
                # Send the tool result back to the model
                messages.append({"role": "tool", "content": str(result), "tool_name": tool_call.function.name})
            else:
                logger.warning(f"No available function for tool call: {tool_call.function.name}")
            
            response = client.chat(model=model, tools=tools, messages=messages, think=True, options=options)


    # final response after all tool calls
    messages.append(response.message)

    print("Final response from model:")
    print(response.message.content)

if __name__ == "__main__":
    main()
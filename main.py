
from ollama import ChatResponse, Client
import logging
import yfinance as yf


# ---- SETUP LOGGING ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def get_stock_price(ticker) -> float:
    stock = yf.Ticker(ticker)
    price = stock.info['regularMarketPrice']
    return price

options = {
    'temperature': 0.7,
    'num_ctx': 2048,
    'top_p': 0.9
}

tools = [
    get_stock_price
]

available_functions = {
    'get_stock_price': get_stock_price
}

def main():
    client = Client()
    model = "qwen3.5:9b"

    user_input = input("Enter your question: ")

    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides stock price information."},
        {"role": "user", "content": user_input}
    ]

    # Stream the response from the model
    # The `think=True` option allows the model to generate intermediate thoughts before providing the final answer.
    response = client.chat(model=model, stream=True, messages=messages, tools=tools, think=True)
    
    # stuff to loop through the response and handle tool calls
    # https://github.com/ollama/ollama-python/blob/main/examples/multi-tool.py
    for chunk in response:
        if chunk.message.thinking:
            logger.info(f"Model is thinking: {chunk.message.content}")
        if chunk.message.content:
            logger.info(f"Model response chunk: {chunk.message.content}")
        if chunk.message.tool_calls:
            for tool_call in chunk.message.tool_calls:
                if function_to_call := available_functions.get(tool_call.function.name):
                    logger.info(f"Calling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
                    result = function_to_call(**tool_call.function.arguments)
                    logger.info(f"Tool result: {result}")
                    # Send the tool result back to the model
                    messages.append(chunk.message)
                    messages.append({"role": "tool", "content": str(result), "tool_name": tool_call.function.name})
                else:
                    logger.warning(f"No available function for tool call: {tool_call.function.name}")
    
    logger.info("------- Sending result back to model")
    if any(msg.get('role') == 'tool' for msg in messages):
        response = client.chat(model=model, stream=True, tools=tools, messages=messages, think=True)
        done_thinking = False
        for chunk in response:
            if chunk.message.thinking:
                logger.info(f"Model is thinking: {chunk.message.content}")
            if chunk.message.content:
                if not done_thinking:
                    logger.info(f"\n ----- Final result:")
                    #print(chunk.message.content, end='', flush=True)
                    done_thinking = True
                print(chunk.message.content, end='', flush=True)
            if chunk.message.tool_calls:
                logger.info(f"Model returned tool calls:")
                logger.info(f"{chunk.message.tool_calls}")



if __name__ == "__main__":
    main()
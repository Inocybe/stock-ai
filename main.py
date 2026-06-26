from openai import OpenAI
from mlx_lm import load, generate

import logging
import server
from tools import define_all_tools

# ---- SETUP LOGGING ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- SETUP OPENAI CLIENT ----
# everything done locally
client = OpenAI(
    base_url="https://localhost:8000/v1",
    api_key="not-needed"
)


def chat():
    model = "models--mlx-community--Qwen3.6-35B-A3B-4bit"
    tls = define_all_tools()
    
    query = input("Enter your question: ")
    messages = [{"role": "user", "content": query}]
    

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tls,
            think=True,
            stream=False,
        )

        msg = response.choices[0].message
        messages.append(msg)

        
        for call in msg.tool_calls:
            if function_to_call := tools.get(call.function.name):
                logger.info(f"Calling tool: {call.function.name} with arguments: {call.function.arguments}")
                result = function_to_call(**call.function.arguments)
                logger.info(f"Tool result: {result}")
                messages.append({"role": "tool", "content": str(result), "tool_name": call.function.name})
            else:
                logger.warning(f"No available function for tool call: {call.function.name}")
        
        if len(msg.tool_calls) == 0:
            print("Final response from model:")
            print(msg.content)
            break




def main():
    chat()



if __name__ == "__main__":
    main()
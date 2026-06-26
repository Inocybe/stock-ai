from openai import OpenAI
from mlx_lm import load, generate

import logging
import server
from tools import define_all_tools, tool_list



# ---- SETUP LOGGING ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Start the server ---
server.start()


# --- SETUP OPENAI CLIENT ----
# everything done locally
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)




def chat():
    model = "mlx-community/Qwen3.6-35B-A3B-4bit"
    tls = define_all_tools()
    
    query = input("Enter your question: ")
    messages = [{"role": "user", "content": query}]
    

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tls,
            stream=False,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            logger.info(f"Final RESPONSE")
            logger.info(f"------------------------------")
            logger.info(f"Assistant response: {msg.content}")
            break

        for call in msg.tool_calls:
            if function_to_call := tool_list.get(call.function.name):
                logger.info(f"Calling tool: {call.function.name} with arguments: {call.function.arguments}")
                result = function_to_call(**call.function.arguments)
                logger.info(f"Tool result: {result}")
                messages.append({"role": "tool", "content": str(result), "name": call.function.name})
            else:
                logger.warning(f"No available function for tool call: {call.function.name}")
        



def main():
    chat()



if __name__ == "__main__":
    main()
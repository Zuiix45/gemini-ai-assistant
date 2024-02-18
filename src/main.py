import asyncio
import llm

from dotenv import load_dotenv

async def main():
    text = input("You: ")
    
    if text != "":
        response = await llm.sendMessage(chat, text)
        print("Assistant: " + response.text)

if __name__ == '__main__':
    load_dotenv()
    
    model = llm.createModel()
    chat = llm.start_chat(model)
    
    asyncio.run(main())
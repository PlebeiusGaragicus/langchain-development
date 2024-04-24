from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


async def main():


    llm = ChatOllama(model="llama3:latest", temperature=0.8)


    prompt = ChatPromptTemplate.from_template(
            template="""You are an human having an informal conversation with a friend.
Your reply should be very short. Don't use proper syntax and punctuation.
Don't be apologetic. Use emoji sparingly.
If I don't say much, don't try to fill in the conversation.

{user_message}
""")

    chain = prompt | llm

    ret = chain.batch(
        [
            {"user_message": "Hey bro what's up?"},
            {"user_message": "Why do birds float on water?"},
            {"user_message": "What's your favorite number?"},
        ]
    )

    for reply in ret:
        print(reply.content)





if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

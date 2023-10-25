import os

from dotenv import load_dotenv
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def load_environment_variables():
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')


def initialize_embeddings():
    return OpenAIEmbeddings(disallowed_special=())


def initialize_deeplake(embeddings):
    return DeepLake(
        dataset_path=os.getenv('DATASET_PATH'),
        read_only=True,
        embedding_function=embeddings,
    )


def initialize_retriever(deep_lake):
    retriever = deep_lake.as_retriever()
    retriever.search_kwargs.update({
        'distance_metric': 'cos',
        'fetch_k': 100,
        'maximal_marginal_relevance': True,
        'k': 10,
    })
    return retriever


def initialize_chat_model():
    return ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name=os.getenv('LANGUAGE_MODEL'), temperature=0.0)


def initialize_conversational_chain(model, retriever):
    return ConversationalRetrievalChain.from_llm(model, retriever=retriever, return_source_documents=True)


def get_user_input():
    question = input("\nПожалуйста, задайте ваш вопрос (или 'quit' чтобы остановить): ")
    return None if question.lower() == 'quit' else question


def main():
    load_environment_variables()
    embeddings = initialize_embeddings()
    deep_lake = initialize_deeplake(embeddings)
    retriever = initialize_retriever(deep_lake)
    model = initialize_chat_model()
    qa = initialize_conversational_chain(model, retriever)

    chat_history = []

    while True:
        question = get_user_input()
        if question is None:
            break

        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))

        first_document = result['source_documents'][0]
        metadata = first_document.metadata
        source = metadata['source']

        print(f"\n\n++source++: {source}")


if __name__ == "__main__":
    main()

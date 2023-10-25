import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings


def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    nav_elements = content.find_all('nav')
    header_elements = content.find_all('header')
    footer_elements = content.find_all('footer')
    for element in nav_elements + header_elements + footer_elements:
        element.decompose()
    return str(content.get_text())


def load_configuration():
    load_dotenv()
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ACTIVELOOP_TOKEN': os.getenv('ACTIVELOOP_TOKEN'),
        'SITE_MAP': os.getenv('SITE_MAP'),
        'DATASET_PATH': os.getenv('DATASET_PATH'),
        'LANGUAGE_MODEL': os.getenv('LANGUAGE_MODEL')
    }


def load_documents(config):
    print('Load pages from sitemap...')
    loader = SitemapLoader(
        config['SITE_MAP'],
        parsing_function=remove_nav_and_header_elements
    )
    return loader.load()


def split_documents(documents):
    print("=" * 100)
    print('Splitting documents...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    text = text_splitter.split_documents(documents)
    print(f'Generated {len(text)} chunks.')
    return text


def create_vector_db(text, config):
    print("=" * 100)
    print('Creating vector DB...')
    embeddings = OpenAIEmbeddings(disallowed_special=())
    deeplake_path = config['DATASET_PATH']
    db = DeepLake(dataset_path=deeplake_path, embedding_function=embeddings, overwrite=True)
    db.add_documents(text)
    print('Vector database updated.')
    return db


def main():
    config = load_configuration()
    os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
    os.environ['ACTIVELOOP_TOKEN'] = config['ACTIVELOOP_TOKEN']

    documents = load_documents(config)
    text = split_documents(documents)
    create_vector_db(text, config)


if __name__ == "__main__":
    main()

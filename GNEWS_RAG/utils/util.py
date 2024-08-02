from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gnews import GNews
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain_core.runnables import RunnablePassthrough
from bs4 import BeautifulSoup
from decouple import config
import os
import feedparser


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)





class VectorDBSelector:
    def __init__(self,db_name,embeddings_model):
        self.db_name = db_name
        self.embeddings_model=embeddings_model
    def load_db_with_docs(self, docs, path):
        if self.db_name == "chroma":
            self.db = Chroma.from_documents(docs, self.embeddings_model,persist_directory=path)
            
        elif self.db_name == "faiss":
            self.db = FAISS.from_documents(docs, self.embeddings_model)
            self.db.save_local(path,index_name="index")
        else:
            raise ValueError(f"Unsupported db_name: {self.db_name}")
    def query_in_db(self,query,k=1):
        if self.db is None:
            raise ValueError("Database is not loaded. Call `load_db_with_docs` first.")
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)
        return retrieved_docs
    def get_retriever(self,k=10):
        if self.db is None:
            raise ValueError("Database is not loaded. Call `load_db_with_docs` first.")
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever
        


async def get_google_news(query, period, max_results):
    urls = []
    google_news = GNews(period=period, max_results=max_results)
    news_results = google_news.get_news(query)
    for news in news_results:
        if 'url' in news.keys():
            urls.append(news['url'])
    parsed_articles = {}
    for i, url in enumerate(urls):
        parsed_articles[i] =await get_parsed_article(url)
    return parsed_articles



async def get_data_from_the_url_manually():
    pass
       
async def get_parsed_article(url):
    print(f"Article for URL {url}: \n")
    google_news = GNews()
    article = google_news.get_full_article(url)

    if article is None:
        print(f"No article found for URL {url}.")
        return None
    
    article.download()
    article.parse()
    article.nlp()
    print( article.text)
    return {
        'title': article.title,
        'text': article.text if article.text else "No Content" ,
        'keywords': article.keywords,
        'keyword_meta': article.meta_keywords,
        'tags': article.tags
    }
    
async def get_parsed_article_for_rss(url):
    print(f"Article for URL {url}: \n")
    google_news = GNews()
    article = google_news.get_full_article(url)

    if article is None:
        print(f"No article found for URL {url}.")
        return None
    
    article.download()
    article.parse()
    article.nlp()
    print( "Text from the rss and article ",article.text)
    return article.text if article.text else "No Content" 
        
    

async def parse_rss(feed_url):
    """extracts the titles,descriptions,source links from the  rss xml"""
    articles=[]
    feed = feedparser.parse(feed_url)

    for i, entry in enumerate(feed.entries):
        title = entry.get('title', 'No title')
        description_html = entry.get('description', 'No description')
        description = BeautifulSoup(description_html, 'html.parser').get_text()

        published = entry.get('published', entry.get('pubDate', 'No published date'))
        link = entry.get('link', 'No link')

        print(f"Title: {title}")
        print(f"Description: {description}")
        print(f"Published Date: {published}")
        print(f"Link: {link}")
        print(i+1)
        articles.append({"Article_No": i + 1, "Title": title, "Description": description, "Published Date": published, "web_text": await get_parsed_article_for_rss(link), "Link": link})
        print("-" * 40)
    return articles
        
        
        
        
 
    


async def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
        
        
async def save_articles(parsed_articles, brand_name, main_folder="content"):
    # Create main folder if it doesn't exist
    await create_folder(main_folder)
    
    # Create a subfolder for the brand
    brand_folder = os.path.join(main_folder, brand_name)
    await create_folder(brand_folder)
    
    # Save each article to a file in the brand's subfolder
    for index, article in parsed_articles.items():
        if article is None:
            print(f"Skipping Article {index + 1} as it is None.")
            continue
        
        file_path = os.path.join(brand_folder, f"Article_{index + 1}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"Title: {article['title']}\n\n")
            file.write(f"Text: {article['text']}\n")
            file.write(f"Keywords: {', '.join(article['keywords'])}\n\n")
            file.write(f"keyword_meta: {article['keyword_meta']}\n\n")
            file.write(f"tags: {article['tags']}\n\n")
    
    return brand_folder


async def save_articles_for_rss(parsed_articles, brand_name, main_folder="content"):
   
    await create_folder(main_folder)
    
   
    brand_folder = os.path.join(main_folder, brand_name)
    await create_folder(brand_folder)
    
    
    for article in parsed_articles:
        if article is None:
            print(f"Skipping Article {article['Article_No']} as it is None.")
            continue
        
        file_path = os.path.join(brand_folder, f"Article_{article['Article_No']}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"Title: {article['Title']}\n\n")
            file.write(f"Description: {article['Description']}\n")
            file.write(f"Published Date: {article['Published Date']}\n")
            file.write(f"Link: {article['Link']}\n")
            file.write(f"Web_text: {article['web_text']}\n")

    return brand_folder






async def are_brand_embeddidngs_already_present(brand_name):
    pass
    

async def  Dir_loader_and_textspliter(brand_folder_path):
    loader = DirectoryLoader(brand_folder_path, glob="**/*.txt", loader_cls=TextLoader,loader_kwargs={'encoding': 'utf-8'} )
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits



# async def create_embeddings(docs,path):
#     embeddings = SentenceTransformerEmbeddings(model_name=config("EMBEDDINGS_MODEL"))
#     db_selector=VectorDBSelector(db_name="chroma",embeddings_model=embeddings)
#     db_selector.load_db_with_docs(docs=docs,path=path)
    
   
   
async def create_embeddings(docs, brand_name, db_name="chroma"):
    embeddings_model = SentenceTransformerEmbeddings(model_name=config("EMBEDDINGS_MODEL"))
    embeddings_folder = f"{db_name}_embeddings"
    brand_embeddings_path = os.path.join(embeddings_folder, brand_name)
    await create_folder(brand_embeddings_path)
    db_selector = VectorDBSelector(db_name=db_name, embeddings_model=embeddings_model)
    db_selector.load_db_with_docs(docs=docs, path=brand_embeddings_path)
    retriever=db_selector.get_retriever(k=10)
    return brand_embeddings_path,retriever




async def get_prompt():
    template="""you will be given some context about the brand and your goal is to perform following things
                1.Analyze Consumer Sentiment
                2.Extract key insights from context
                3.Detect emerging crises or negative news from the context
                4.Extract the trending topics about the brand
                Note: Do not use your external knowledge ,only use the information provided to you
                also consider that you have to do influencer marketing for this brand, what type of campaign will you do , to boost the brand's value given the details about brand and it's current social listening state
                5.suggest some campains we can do with this brand
                6.also tell in which article the information is present
                context:{context}
                question:{question}
                """
    prompt=PromptTemplate.from_template(template)
    return prompt       



async def initiate_llm(llm,retriever,prompt,query):
    
    rag_chain = (
    {"context": retriever |  format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
    
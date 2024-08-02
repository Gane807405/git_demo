from fastapi import FastAPI
from schemas.schemas import GnewsRequestModel
from langchain_community.llms import Ollama
from utils.util import (get_google_news,
                        save_articles,
                        Dir_loader_and_textspliter,
                        create_embeddings,
                        initiate_llm,
                        get_prompt,
                        parse_rss,
                        save_articles_for_rss)
from decouple import config
app =FastAPI()





@app.post("/gnews/social_listening/")

async def RAG(request:GnewsRequestModel):
    req_obj=request.model_dump()
    
    if req_obj["rss_or_gnews"] == "gnews":
        # are_brand_embeddidngs_already_present(brand_name=req_obj["brand_name"])
        parsed_articles=await get_google_news(query=req_obj["query"],period=config("PERIOD"),max_results=int(config("MAX_RESULTS")))
        brand_folder_path=await save_articles(parsed_articles,brand_name=req_obj["brand_name"])
        

    
    else :
        parsed_articles= await parse_rss(req_obj["rss_link"])
        print(type(parsed_articles))
        brand_folder_path=await save_articles_for_rss(parsed_articles=parsed_articles,brand_name=req_obj["brand_name"])
        

        
    docs=await Dir_loader_and_textspliter(brand_folder_path=brand_folder_path)
    brand_embeddings_folder_path ,retriever=await create_embeddings(docs=docs, brand_name=req_obj["brand_name"], db_name="chroma")
    llm = Ollama(model=config("LLM"))
    prompt=await get_prompt()
    await initiate_llm(llm=llm,retriever=retriever,prompt=prompt,query=req_obj["query"])

        
        
    
    
    
    
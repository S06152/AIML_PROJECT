
from langchain.prompts import PromptTemplate
from src.retriever import retrieve_docs
from src.config import MODEL_PROVIDER, OPENAI_API_KEY

# Hugging Face imports
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline

# OpenAI imports
from langchain_openai import ChatOpenAI


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful travel assistant.
Use the following travel guide context to answer the question.
If the answer is not found, say you don't know — don’t make it up.

Context:
{context}

Question:
{question}

Answer:"""
)

if MODEL_PROVIDER == "huggingface":
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
elif MODEL_PROVIDER == "openai":
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.7)
else:
    raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")


qa_chain = prompt_template | llm

def generate_answer(query: str) -> str:
    docs = retrieve_docs(query)
    context = "\n".join(docs)
    result = qa_chain.invoke({"context": context, "question": query})
    # If using OpenAI, return only the 'content' field
    if MODEL_PROVIDER == "openai":
        #print(result.content)
        return result.content
    return result

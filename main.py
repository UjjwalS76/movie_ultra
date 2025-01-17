import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo, get_query_constructor_prompt
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

# Set up environment variable
# os.environ["OPENAI_API_KEY"] = "pplx-your-perplexity-api-key"  # Removed in favor of Streamlit secrets

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                              openai_api_key=st.secrets["PERPLEXITY_API_KEY"],
                              openai_api_base="https://api.perplexity.ai")

# Sample documents
docs = [
    Document(
        page_content="A poor but big-hearted man takes orphans into his home. After discovering his scientist father's invisibility device, he rises to the occasion and fights to save his children and all of India from the clutches of a greedy gangster",
        metadata={"year": 2006, "director": "Rakesh Roshan", "rating": 7.1, "genre": "science fiction"},
    ),
    Document(
        page_content="The story of six young Indians who assist an English woman to film a documentary on the freedom fighters from their past, and the events that lead them to relive the long-forgotten saga of freedom",
        metadata={"year": 2006, "director": "Rakeysh Omprakash Mehra", "rating": 9.1, "genre": "drama"},
    ),
    Document(
        page_content="A depressed wealthy businessman finds his life changing after he meets a spunky and care-free young woman",
        metadata={"year": 2007, "director": "Anurag Basu", "rating": 6.8, "genre": "romance"},
    ),
    Document(
        page_content="A schoolteacher's world turns upside down when he realizes that his former student, who is now a world-famous artist, may have plagiarized his work",
        metadata={"year": 2023, "director": "R. Balki", "rating": 7.8, "genre": "drama"},
    ),
    Document(
        page_content="A man returns to his country in order to marry his childhood sweetheart and proceeds to create misunderstanding between the families",
        metadata={"year": 1995, "director": "Aditya Chopra", "rating": 8.1, "genre": "romance"},
    ),
    Document(
        page_content="The story of an Indian army officer guarding a picket alone in the Kargil conflict between India and Pakistan",
        metadata={"year": 2003, "director": "J.P. Dutta", "rating": 7.9, "genre": "war"},
    ),
    Document(
        page_content="Three young men from different parts of India arrive in Mumbai, seeking fame and fortune",
        metadata={"year": 1975, "director": "Ramesh Sippy", "rating": 8.2, "genre": "action"},
    ),
    Document(
        page_content="A simple man from a village falls in love with his new neighbor. He enlists the help of his musical-theater friends to woo the lovely girl-next-door away from her music teacher",
        metadata={"year": 1990, "director": "Sooraj Barjatya", "rating": 7.7, "genre": "musical"},
    ),
    Document(
        page_content="A young mute girl from Pakistan loses herself in India with no way to head back. A devoted man undertakes the task to get her back to her homeland and unite her with her family",
        metadata={"year": 2015, "director": "Kabir Khan", "rating": 8.0, "genre": "drama"},
    ),
    Document(
        page_content="Three idiots embark on a quest for a lost buddy. This journey takes them on a hilarious and meaningful adventure through memory lane and gives them a chance to relive their college days",
        metadata={"year": 2009, "director": "Rajkumar Hirani", "rating": 9.4, "genre": "comedy"},
    ),
]

# Create vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Define metadata fields for self-querying
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie.",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"

# Initialize LLM
llm = ChatOpenAI(temperature=0,
                 model="llama-3.1-sonar-small-128k-online",
                 openai_api_key=st.secrets["PERPLEXITY_API_KEY"],
                 openai_api_base="https://api.perplexity.ai")

# Create self-querying retriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)


# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Streamlit app
st.title("Movie Recommendation System")

# User input
query = st.text_input("Enter your movie preference:")

if st.button("Search"):
    if query:
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)
        
        # RAG prompt
        prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
        {context}

        Question: {question}
        """)
        # RAG chain 
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Display results
        with st.spinner("Searching..."):
            result = chain.invoke(query)

        st.subheader("Recommended Movies:")
        st.write(result)

    else:
        st.warning("Please enter a query.")

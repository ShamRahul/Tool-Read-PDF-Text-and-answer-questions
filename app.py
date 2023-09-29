# Import necessary libraries and modules
import streamlit as st  # Streamlit library for creating web apps
from langchain.llms import OpenAI  # LangChain module for interacting with OpenAI models
from langchain.text_splitter import CharacterTextSplitter  # LangChain module for splitting text into chunks
from langchain.embeddings import OpenAIEmbeddings  # LangChain module for creating embeddings using OpenAI models
from langchain.vectorstores import Chroma  # LangChain module for creating a vector store from documents
from langchain.chains import RetrievalQA  # LangChain module for creating a QA chain
import fitz  # PyMuPDF library for reading PDF files

# Function to read the uploaded file
def read_file(uploaded_file):
    # Check the file type of the uploaded file
    if uploaded_file.type == 'application/pdf':
        # Initialize a PyMuPDF document object to read the PDF file
        pdf_document = fitz.open(stream=uploaded_file.getvalue())
        document_text = ""  # Initialize an empty string to store the extracted text
        # Iterate over each page in the PDF and extract text
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)  # Load each page
            document_text += page.get_text("text")  # Extract text from the page and append it to the document_text string
        pdf_document.close()  # Close the PDF document
        return document_text  # Return the extracted text
    elif uploaded_file.type == 'text/plain':
        # If the uploaded file is a TXT file, read and decode the file content and return it
        return uploaded_file.read().decode()
    else:
        # Display an error message if the uploaded file type is unsupported and return None
        st.error('Unsupported file type')
        return None

# Function to generate a response based on the uploaded file and user's query
def generate_response(uploaded_file, openai_api_key, query_text):
    # Read the uploaded file using the read_file function
    document_text = read_file(uploaded_file)
    if document_text is not None:  # Proceed if the document_text is not None
        # Initialize a CharacterTextSplitter object to split the document_text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # Create documents from the chunks
        texts = text_splitter.create_documents([document_text])
        # Initialize an OpenAIEmbeddings object to create embeddings using the provided OpenAI API key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a Chroma vector store from the documents and embeddings
        db = Chroma.from_documents(texts, embeddings)
        # Create a retriever interface from the Chroma vector store. The retriever interface is created from the Chroma vector store, which is essentially a store of vector representations (embeddings) of the chunks of text from the uploaded document. The retriever's role is to efficiently search through these vector representations to find the chunks of text that are most relevant to a given query
        retriever = db.as_retriever()
        # Initialize a RetrievalQA object to create a QA chain using the OpenAI model and the retriever. RetrievalQA is responsible for creating a Question-Answering chain using the OpenAI model and the retriever. It takes a user’s query, uses the retriever to find the most relevant chunks of text from the uploaded document, and then uses the OpenAI model to generate a response based on those relevant chunks.
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        # Run the QA chain on the user's query and return the result
        return qa.run(query_text)

# Set the page title and display the app title
st.set_page_config(page_title='🦜🔗 PDF or TXT File Reader')
st.title('🦜🔗 PDF or TXT File Reader')

# Create a file uploader widget to allow the user to upload a PDF or TXT file
uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf'])
# Create a text input widget to allow the user to enter their question
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Initialize a list to store the result and create a form for user input
result = []
with st.form('myform', clear_on_submit=True):
    # Create a password input widget to allow the user to enter their OpenAI API key
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    # Create a submit button for the form
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    # If the form is submitted and the OpenAI API key is valid, generate a response
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)  # Append the response to the result list
            del openai_api_key  # Delete the OpenAI API key after use

# If there is a result, display it in an info box
if len(result):
    st.info(response)


# PDF CHATBOT [Application](https://huggingface.co/spaces/ashish-001/PDF-QA-CHATBOT)
This project implements a Retrieval-Augmented Generation (RAG) system designed to enhance query responses by leveraging advanced natural language understanding and efficient data retrieval mechanisms. The system integrates GEMINI as the language model and FAISS as the vector store, streamlined using LangChain to ensure seamless and optimized performance.

### Key Features

- **Advanced Language Understanding**: Utilizes GEMINI to provide sophisticated natural language understanding and generation capabilities.
- **Efficient Data Retrieval**: Integrates FAISS for storing and retrieving vector data quickly and accurately.
- **Streamlined Pipeline**: Employs LangChain for seamless integration of the language model and vector store, optimizing the overall system performance.


## Steps to run the program on Windows
1. Create a virtual environment 
```
python -m venv "environment name"
```
2. Activate the virtual environment
```
"environment name"\Scripts\activate
```
3. Install all required libraries
```
pip install -r requirements.txt
```
4. Set the api key in an .env file
```
GOOGLE_API_KEY = ''  # insert gemini api key here
```
5. Run the program
```
streamlit run application.py
```

## Images
![Alt text](img.png)

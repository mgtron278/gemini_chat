

---

# Chat PDF with Gemini LLM - Surya Guttikonda.

This project allows users to interact with PDF documents using Gemini Large Language Model (LLM) through a Streamlit interface. Users can upload PDF files and ask questions related to the content of the PDF, and the system will provide answers based on the context of the documents.

## How to Use

1. Clone this repository to your local machines

   ```bash
   git clone https://github.com/your_username/your_repository.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain your Google API Key and set it as an environment variable. You can refer to the [Google Cloud documentation](https://cloud.google.com/docs/authentication/api-keys) for instructions on how to get an API Key.

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Once the app is running, you can upload your PDF files using the file uploader.

6. Ask questions related to the content of the PDF files in the text input field provided.

7. The system will process your question and provide an answer based on the context of the uploaded PDF files.

## Dependencies

- `streamlit`: For creating the web application interface.
- `PyPDF2`: For reading PDF files.
- `langchain`: For text processing and question answering capabilities.
- `langchain_google_genai`: For Google Generative AI embeddings.
- `google.generativeai`: For accessing Google's Generative AI models.
- `dotenv`: For loading environment variables from a .env file.

## Configuration

Make sure to set your Google API Key as an environment variable named `GOOGLE_API_KEY`. You can do this by creating a `.env` file in the root directory of the project and adding the following line:

```
GOOGLE_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Google API Key.

## About

This project is built using Python and Streamlit, leveraging Gemini LLM and Google's Generative AI models for text processing and question answering tasks.

---

# split-documents-upload-to-pinecone

Divide documents and upload text segments to Pinecone with the script.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/kirill-markin/split-documents-upload-to-pinecone.git
    ```

2. Install the required packages:

    ```bash
    cd split-documents-upload-to-pinecone
    pip install -r requirements.txt
    ```

3. Set up environment variables in a `.env` file in the project root based on the `.env.example` file:

    ```ini
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENVIRONMENT=your_pinecone_environment
    PINECONE_INDEX_NAME=your_index_name
    ```

    Replace `your_pinecone_api_key` and `your_pinecone_environment` with your actual values.

    `your_pinecone_environment` — the name of the Pinecone environment you want to use. For example `us-central1-gcp`.
    `your_index_name` — the name of the index you want to create. You can use any name.

## Usage

Add documents to the `data` folder. It can be a single file or multiple files in inner folders. The script will process all files `*.md` in the folder.

Run the script:

```bash
python3 main.py
```

## License

This project is licensed under the [MIT License](LICENSE).

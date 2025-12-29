```bash
pip install -r requirements.txt
```

## Usage

1. Make sure Ollama is running locally (e.g., `ollama serve`)
2. Pull the Qwen2.5 model if not already present:
   ```bash
   ollama pull qwen2.5
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run tmp.py
   ```

## Features

- **Natural Language to SQL**: Ask questions in Hindi/English and get SQL queries
- **Schema Awareness**: AI understands your database structure
- **Safety**: Prevents destructive queries with LIMIT enforcement
- **Fast Results**: Direct PostgreSQL connection with caching

```
GOOGLE_API_KEY = "API Key"
GEMINI_MODEL = "gemini-2.0-flash"

[connections.postgresql_volume]
dialect = "postgresql"
username = "quorbit"
password = "quorbit"
host = "127.0.0.1"
port = "15432"
database = "volume"

[connections.postgresql_quorbit]
dialect = "postgresql"
username = "quorbit"
password = "quorbit"
host = "127.0.0.1"
port = "15432"
database = "quorbit"

```

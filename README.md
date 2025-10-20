# Multi-Modal RAG Q&A System

A comprehensive Retrieval-Augmented Generation (RAG) system supporting text, images, audio, and URLs with advanced multimodal capabilities powered by Ollama's local LLMs.

## 🚀 Features

- **Multi-Modal Support**: Process text documents, images (with OCR), audio files, and web URLs
- **Local LLM Processing**: Privacy-preserving inference using Ollama (Llama 3, Llama 3.2 Vision, LLaVA)
- **Advanced RAG Pipeline**: Intelligent document retrieval and context-aware response generation
- **Real-time Streaming**: Streaming chat responses for better user experience
- **OCR Integration**: Extract text from images using Tesseract OCR
- **Audio Transcription**: Transcribe audio files using Whisper (via FFmpeg)
- **Knowledge Base Management**: Upload, index, and query multiple document types
- **Flexible Architecture**: FastAPI backend with Streamlit frontend
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🏗️ Architecture

```
Multi-Modal-RAG-Q-A/
├── backend/              # FastAPI application
│   ├── core/            # Core RAG functionality
│   │   ├── config.py           # Configuration management
│   │   ├── logger.py           # Logging utilities
│   │   ├── embedding_manager.py # Vector embeddings
│   │   ├── model_selector.py   # LLM model selection
│   │   ├── rag_engine.py       # RAG pipeline
│   │   └── text_extractor.py   # Document processing
│   ├── routes/          # API endpoints
│   │   ├── chat_stream.py      # Streaming chat
│   │   ├── file_chat.py        # File-based chat
│   │   ├── knowledge_base.py   # KB management
│   │   ├── ocr_image.py        # Image OCR
│   │   ├── summarize.py        # Text summarization
│   │   ├── transcribe_audio.py # Audio processing       (Future Enhancement)
│   │   ├── url_chat.py         # URL-based chat         (Future Enhancement)
│   │   ├── retriever_routes.py # Document retrieval
│   │   └── inference_routes.py # LLM inference
│   ├── app.py           # FastAPI main application
│   ├── requirements.txt # Python dependencies
│   └── .env.example     # Environment variables template
├── frontend/            # Streamlit UI
│   ├── app_ui.py       # Main UI application
│   └── requirements.txt # UI dependencies
├── data/               # Document storage               (Auto generated file)
└── logs/               # Application logs               (Auto generated file)
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **Disk Space**: At least 10GB for models and data

### Required Software
1. **Ollama**: Local LLM runtime
2. **Tesseract OCR**: For image text extraction
3. **FFmpeg**: For audio processing
4. **Git**: For cloning the repository

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Multi-Modal-RAG-Q-A.git
cd Multi-Modal-RAG-Q-A
```

### 2. Install Ollama

**Windows/macOS:**
```bash
# Download and install from https://ollama.ai/
curl -fsSL https://ollama.ai/install.sh | sh
```

**Pull Required Models:**
```bash
ollama pull llama3:latest
ollama pull llama3.2-vision
ollama pull llava-llama3
ollama pull nomic-embed-text
```

**Start Ollama Server:**
```bash
ollama serve
```

### 3. Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Add to system PATH

### 4. Install FFmpeg

**Windows:**
1. Download from: https://ffmpeg.org/download.html
2. Extract and add to system PATH

### 5. Set Up Python Environment

**Install Backend Dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

**Install Frontend Dependencies:**
```bash
cd frontend
pip install -r requirements.txt
```

## 🚀 Running the Application

### 1. Start the Backend Server

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

The FastAPI server will start at: `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

### 2. Start the Frontend UI

Open a new terminal:

```bash
cd frontend
streamlit run app_ui.py
```

The Streamlit interface will open at: `http://localhost:8501`

### 3. Verify Installation

Check that all services are running:

- **Ollama**: `http://localhost:11434`
- **Backend API**: `http://localhost:8000/docs`
- **Frontend UI**: `http://localhost:8501`

## 📚 Usage Guide

### 1. Knowledge Base Management

**Upload Documents:**
- Navigate to "Knowledge Base" section
- Supported formats: PDF, TXT, DOCX, MD
- Upload single or multiple files
- System automatically processes and indexes documents

**Upload Images (with OCR):**
- Navigate to "Image OCR" section
- Supported formats: JPG, PNG, BMP, TIFF
- System extracts text using Tesseract
- Extracted text is added to knowledge base

**Upload Audio Files:**
- Navigate to "Audio Transcription" section
- Supported formats: MP3, WAV, M4A, OGG
- System transcribes using Whisper
- Transcripts are indexed for search

### 2. Chat with Documents

**File-Based Chat:**
1. Upload documents to knowledge base
2. Navigate to "File Chat" section
3. Ask questions about your documents
4. System retrieves relevant context and generates answers

**URL-Based Chat:**
1. Navigate to "URL Chat" section
2. Enter website URL
3. System extracts content from the page
4. Ask questions about the webpage content

### 3. Streaming Chat

**Real-time Responses:**
1. Navigate to "Chat Stream" section
2. Type your question
3. Receive streaming responses in real-time
4. Uses RAG context from knowledge base

### 4. Text Summarization

**Summarize Documents:**
1. Upload or paste text content
2. Navigate to "Summarize" section
3. Choose summarization mode (bullet points, paragraph, key points)
4. Get AI-generated summaries

### 5. Advanced Features

**Model Selection:**
- Switch between Llama3, Llama 3.2 Vision, and LLaVA
- Automatic model selection based on input type
- Configure temperature and max tokens

**Retrieval Configuration:**
- Adjust chunk size for document processing
- Configure overlap for better context
- Set top-k results for retrieval

## 🔒 Security Considerations

1. **Local Processing**: All data processed locally - no external API calls
2. **Data Privacy**: Documents stored locally in `data/` directory
3. **Access Control**: Implement authentication for production deployments (Future Enhancement)
4. **Input Validation**: All inputs validated and sanitized
5. **CORS Configuration**: Configure for production environments

## 🚧 Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Check if Ollama is running
curl http://localhost:11434

# Restart Ollama
ollama serve
```

**2. Tesseract Not Found**
```bash
# Windows: Check PATH variable
echo %PATH%

# Verify installation
tesseract --version
```

**3. FFmpeg Error**
```bash
# Verify installation
ffmpeg -version

# Add to PATH if missing
```

**4. Memory Issues**
```bash
# Reduce chunk size in .env
CHUNK_SIZE=500

# Use smaller models
OLLAMA_MODEL=llama3:8b
```

**5. Port Already in Use**
```bash
# Windows: Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:8000 | xargs kill -9
```

### Debug Mode

Enable detailed logging:

```bash
# In backend/.env
LOG_LEVEL=DEBUG
```

## 🔄 Updates and Maintenance

### Update Ollama Models

```bash
ollama pull llama3:latest
ollama pull llama3.2-vision
ollama pull llava-llama3
```

### Update Python Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt --upgrade

# Frontend
cd frontend
pip install -r requirements.txt --upgrade
```

### Clean Up Data

```bash
# Remove indexed documents
rm -rf data/vector_store

# Clear logs
rm -rf logs/*.log
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama**: Local LLM inference engine
- **LangChain**: RAG framework and document processing
- **FastAPI**: High-performance web framework
- **Streamlit**: Interactive UI framework
- **Tesseract OCR**: Open-source OCR(Optical Character Recognition) engine
- **FFmpeg**: Multimedia processing framework
- **Meta AI**: Llama 3 language models

## 🎓 Learning Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Tutorial](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---
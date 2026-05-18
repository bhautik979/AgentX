#!/bin/bash
# setup.sh - Quick setup script

echo "🚀 Financial RAG System Setup"
echo "=============================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📥 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

# Create .env file
echo "⚙️  Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file (UPDATE WITH YOUR PINECONE_API_KEY)"
else
    echo "⚠️  .env file already exists"
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p core api/routes tests data/reports logs

# Create __init__.py files
touch core/__init__.py
touch api/__init__.py
touch api/routes/__init__.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 NEXT STEPS:"
echo "1. Update .env file with your Pinecone API key"
echo "2. Start Ollama: ollama run mistral (in another terminal)"
echo "3. Run the server: python main.py"
echo "4. Visit: http://localhost:8000/docs"
echo ""
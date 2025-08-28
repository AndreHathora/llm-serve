#!/bin/bash

echo "LLM Latency Comprehensive Analysis"
echo "================================="

if [ ! -f "scripts/baseurl.txt" ]; then
    echo "Error: scripts/baseurl.txt not found"
    echo "Please create this file with your Hathora service URL"
    echo "Example: echo 'your-service.edge.hathora.dev:port' > scripts/baseurl.txt"
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not detected. Activating..."
    source venv/bin/activate
fi

echo "Installing dependencies..."
pip install -q requests numpy pandas matplotlib seaborn

echo "Starting comprehensive latency analysis..."
cd tests
python comprehensive_latency_analyzer.py

if [ -f "../data/enhanced_latency_results.csv" ]; then
    echo "Analysis complete!"
    echo "Results saved to:"
    echo "   - data/enhanced_latency_results.csv"
    echo "   - data/enhanced_latency_analysis.png"
    echo ""
    echo "Use this data for your blog post analysis."
else
    echo "Analysis failed or no results generated"
    exit 1
fi

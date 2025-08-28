#!/bin/bash

echo "LLM Latency Profiling"
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

echo "Starting full profile latency analysis..."
cd src
python fullprofile.py

if [ -f "../data/full_profile_latency_results.csv" ]; then
    echo "Analysis complete!"
    echo "Results saved to:"
    echo "   - data/full_profile_latency_results.csv"
    echo "   - data/full_profile_latency_analysis.png"
    echo ""
    echo "Use this data for your blog post analysis."
else
    echo "Analysis failed or no results generated"
    exit 1
fi

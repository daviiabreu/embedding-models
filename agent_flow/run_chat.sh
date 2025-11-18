#!/bin/bash
# Launcher script for the Inteli Robot Dog Tour Guide chat interface

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the agent_flow directory
cd "$SCRIPT_DIR" || exit 1

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required environment variables are set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  Warning: GOOGLE_API_KEY not found in environment"
    echo "Loading from .env file..."
fi

# Run the chat interface
echo "Starting Inteli Robot Dog Tour Guide..."
echo ""

python3 chat.py "$@"

#!/bin/bash
# Setup script for Stampede Alert System React Frontend

echo "🚀 Stampede Alert System - React Frontend Setup"
echo "==============================================="
echo ""

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js from https://nodejs.org"
    exit 1
fi

echo "✓ Node.js and npm found"
echo "  - Node version: $(node --version)"
echo "  - npm version: $(npm --version)"
echo ""

# Navigate to Frontend directory
cd "$(dirname "$0")" || exit

echo "📦 Installing dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
    echo ""
    echo "✨ Setup complete!"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start development server:"
    echo "   npm run dev"
    echo ""
    echo "2. Build for production:"
    echo "   npm run build"
    echo ""
    echo "3. Preview production build:"
    echo "   npm run preview"
    echo ""
    echo "📚 Documentation:"
    echo "   - README-REACT.md           - Feature documentation"
    echo "   - QUICKSTART-REACT.md       - 5-minute quick start"
    echo "   - PROJECT-OVERVIEW.md       - Complete overview"
    echo "   - MIGRATION-SUMMARY.md      - What was built"
    echo ""
    echo "Happy coding! 🎉"
else
    echo "❌ Installation failed. Please check your internet connection and try again."
    exit 1
fi

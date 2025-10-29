#!/bin/bash
# Setup script for Multimodal Learning Jupyter Notebooks
# Author: Kai Guo (guokai8@gmail.com)

echo "🚀 Setting up Multimodal Learning Jupyter Environment..."

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "📍 Python version: $python_version"

if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]; then
    echo "❌ Error: Python 3.8 or higher required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv multimodal_env
source multimodal_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No GPU detected, installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing other requirements..."
pip install -r requirements.txt

# Install Jupyter extensions
echo "🔧 Setting up Jupyter extensions..."
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import transformers
import matplotlib
import numpy
import jupyter
print('✅ All core packages installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers version: {transformers.__version__}')
"

# Create launch script
cat > launch_notebooks.sh << 'EOF'
#!/bin/bash
echo "🚀 Launching Multimodal Learning Notebooks..."
source multimodal_env/bin/activate
echo "📖 Starting Jupyter Lab..."
jupyter lab --notebook-dir=. --ip=0.0.0.0 --port=8888 --no-browser
EOF

chmod +x launch_notebooks.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate environment: source multimodal_env/bin/activate"
echo "2. Start Jupyter: ./launch_notebooks.sh"
echo "   OR: jupyter lab"
echo "3. Open index.ipynb to begin learning!"
echo ""
echo "📚 Happy learning with multimodal AI! 🚀"

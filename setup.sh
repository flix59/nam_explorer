#!/bin/bash

# CONFIG
PROJECT_NAME="nam_explorer"
ENV_NAME="nam"
PYTHON_VERSION="3.13.5"

# 1. Create project structure
# echo "📁 Creating project folder: $PROJECT_NAME"
# mkdir -p $PROJECT_NAME/{data,notebooks,src,tests}

# cd $PROJECT_NAME || exit 1

# 2. Create Conda environment
echo "🐍 Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
# conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 3. Activate Conda environment
echo "✅ Activating conda environment: $ENV_NAME"
# You can't "conda activate" in a non-interactive script, so use `conda run`
# For the rest of the script, we'll prefix commands with `conda run`

# 4. Install uv via pip inside Conda
echo "⚡ Installing uv inside Conda environment..."
conda run -n $ENV_NAME pip install uv

# 5. Install dependencies via uv
echo "📦 Installing Python packages with uv..."
conda run -n $ENV_NAME uv pip install numpy pandas scikit-learn matplotlib jupyter wandb

# 6. Generate requirements.txt (optional)
# echo "📝 Generating requirements.txt..."
# conda run -n $ENV_NAME pip freeze > requirements.txt

# 7. Export conda environment
echo "📄 Exporting conda environment to environment.yml..."
conda env export --name $ENV_NAME > environment.yml

# 8. Create base Python files
echo "🧱 Creating project files..."
touch README.md paper_notes.md
touch src/__init__.py src/main.py src/utils.py
touch tests/__init__.py tests/test_utils.py

# 9. Done
echo "✅ Setup complete!"
echo "👉 To start working:"
echo "   conda activate $ENV_NAME"
echo "   uv pip install <new-package>  # if needed"

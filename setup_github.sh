#!/bin/bash

# Create a .gitignore file to exclude unnecessary files
echo "Creating .gitignore..."
cat <<'EOF' > .gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
pip-selfcheck.json
*.egg-info/
.env

# Node.js / Frontend (Vite)
web/visual-recognition-frontend/node_modules/
web/visual-recognition-frontend/dist/
web/visual-recognition-frontend/.env*
web/visual-recognition-frontend/.vite/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# macOS
.DS_Store
EOF

# Initialize git repository
git init

# Remove nested .git directory from frontend if it exists
if [ -d "web/visual-recognition-frontend/.git" ]; then
  echo "Removing nested .git directory from frontend..."
  rm -rf web/visual-recognition-frontend/.git
fi

# Add all files (respecting .gitignore)
git add .

# Initial commit
git commit -m "Initial commit: Visual Recognition System"

# Display status
git status

# Instructions for remote repository
echo "
GitHub repository setup complete.

To push this to your GitHub repository, follow these steps:

1. Create a new repository on GitHub.
2. Run the following commands:

   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
"

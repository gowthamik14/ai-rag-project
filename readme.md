Open Terminal
     │
     ▼
Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     │
     ▼
Add to PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
     │
     ▼
Verify brew
brew --version
     │
     ▼
Install Ollama
brew install ollama
     │
     ▼
Start Ollama
brew services start ollama
     │
     ▼
Pull Model
ollama pull qwen2.5:7b
     │
     ▼
Test
ollama run qwen2.5:7b "Hello"
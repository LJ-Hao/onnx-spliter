PYTHON = python3
PIP = pip3
VENV_DIR = .env
REQUIREMENTS = requirements.txt
SRC_DIR = src

all: install convert

install: $(REQUIREMENTS)
	@echo "Setting up virtual environment and installing dependencies..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@$(VENV_DIR)/bin/$(PIP) install -r $(REQUIREMENTS)
	@sudo apt update && sudo apt upgrade -y
	@sudo apt install libgl1-mesa-glx -y

test:
	@echo "Running tests..."
	@$(VENV_DIR)/bin/pytest $(TEST_DIR)

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_DIR)
	@rm -f $(ONNX_FILE)

install-deps:
	@echo "Installing project dependencies..."
	@$(PIP) install -r $(REQUIREMENTS)

run:
	@echo "Running the application..."
	@$(VENV_DIR)/bin/$(PYTHON) main.py

help:
	@echo "Makefile Commands:"
	@echo "  make install    - Set up virtual environment and install dependencies"
	@echo "  make test       - Run tests using pytest"
	@echo "  make clean      - Clean up generated files"
	@echo "  make run        - Run the main application"
	@echo "  make help       - Display this help message"

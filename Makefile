PYTHON = python3
PIP = pip3
VENV_DIR = .env
REQUIREMENTS = requirements.txt
SRC_DIR = src
TEST_DIR = tests

all: install convert

test:
	$(PYTHON) convert.py --model=./yolo11n.onnx --node=/model.2/Split

install: $(REQUIREMENTS)
	@echo "Setting up virtual environment and installing dependencies..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@sudo apt update && sudo apt upgrade -y
	@$(VENV_DIR)/bin/$(PIP) install -r $(REQUIREMENTS)
	@echo "Virtual environment created. Use 'source $(VENV_DIR)/bin/activate' to activate the environment."


clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_DIR)
	@rm -f $(ONNX_FILE)


help:
	@echo "Makefile Commands:"
	@echo "  make install    - Set up virtual environment and install dependencies"
	@echo "  make test       - Test spliter to split yolo11n"
	@echo "  make clean      - Clean up generated files"
	@echo "  make help       - Display this help message"

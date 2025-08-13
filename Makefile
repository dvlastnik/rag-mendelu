PYTHON=python3

.PHONY: all etl main clean

all: etl main

etl:
	@echo "Running ETL pipeline and inserting data to main database (ChromaDB)..."
	${PYTHON} run_etl.py

main:
	@echo "Asking LLM with RAG..."
	${PYTHON} main.py

clean:
	@echo "Cleaning databases..."
# 	TODO
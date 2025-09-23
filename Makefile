PYTHON=python3

.PHONY: all etl main clean

all: etl main

etl:
	@echo "Running ETL pipeline and inserting data to main database (ChromaDB)..."
	${PYTHON} main.py --run-etl

check-dbs:
	@echo "Checking all databases if there are some data inside"
	${PYTHON} main.py --check-dbs

main:
	@echo "Asking LLM with RAG..."
	${PYTHON} main.py

clean:
	@echo "Cleaning databases..."
# 	TODO
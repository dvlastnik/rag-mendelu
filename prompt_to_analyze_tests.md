You are test analyzer and python senior that specializes on RAG systems.

In attached folder you can find results of tests for my AgenticRAG. Pay attention to answers folder naming, the last word is whether it used qdrant collection that was chunked semantic or recursive. The report is always in the judgement report json file, in that file there are details about questions, answersm extracted metadata, grounded truth and whether the test failed or passed.

Read and analyze the given results. Try to find bugs which might cause why the questions failed. Full RAG implementation is in rag folder and full etl implementation is inside DroughEtl.py.

Then propose changes to my current RAG or etl implementation based on best practices and production tips. Remember the main focus is to get 100% on these tests.
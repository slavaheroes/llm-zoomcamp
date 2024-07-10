# 03 Vector Search Notes

### Introduction to Vector Search

**Multi-modal data** (images, documents, etc) are represented as vectors. When a query is given, a database finds the nearest neighbors to the query vector representation. 

### Semantic Search with Elastic search

1. Generate embeddings using pre-trained Deep learning models

    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    encoded_vector = model.encode(text)
    ```

    Now, one data instance looks like this: 
    ```python
    {
        'text': 'GitHub - DataTalksClub data-engineering-zoomcamp#prerequisites',
        'section': 'General course-related questions',
        'question': 'Course - What are the prerequisites for this course?',
        'course': 'data-engineering-zoomcamp',
        'text_vector': encoded_vector
    }
    ```

2. Add documents into the Elastic Search index
    ```python
    for doc in documents:
        es_client.index(index=index_name, document=doc)
    ```
3. Example of Search:
    ```python
    query = {
        "field": "text_vector",
        "query_vector": model.encode(my_text_query),
        "k": 5,
        "num_candidates": 10000, 
    }
    res = es_client.search(index=index_name, 
        knn=query, source=["text", "section", "question", "course"])

    print(res["hits"]["hits"])
    ```

## Evaluation of Retrieval

### Metrics  

- Mean Reciprocal Rank (MRR): $MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i} $
- Hit Rate (HR) or Recall at k: $HR@k = \frac{\text{Number of queries with at least one relevant document in top k}}{|Q|}$
- etc.

### Ground Truth Dataset Generation

For each answer in the FAQ, we need to generate 5 questions that shouldn't contain similar words. We do that using LLM.
```python
for record in documents:
    prompt = make_prompt(record)
    questions = llm(prompt)
```

We can save it into `pd.DataFrame` with columns `question, course, document_id`. (`document_id` $\sim$ answer)

### Evaluation

```python
relevance = []
for question in ground_truth:
    doc_id = question['id']
    results = retrieval(question['question'])
    
    relevance.append(
        [d['id'] == doc_id for d in results]
    )

mrr, hit_rate = compute_metrics(relevance)
```

`retrieval` is a function that we want to evaluate. The best one should be chosen based on the evaluation metrics.
# Pinecone & Vector Database Flashcards

## Concept: Vector Database
**Question:** What is a Vector Database (like Pinecone)?
**Answer:** A specialized database designed to store "vectors" (lists of numbers) and perform "similarity searches" rather than exact keyword matches. It answers "How close is A to B?" instead of "Does A equal B?".

## Concept: Index
**Question:** What is an Index in Pinecone?
**Answer:** The highest-level container for your data. It is similar to a "Database" or "Table" in SQL. You must define its configuration (Dimension, Metric, Cloud Region) when creating it, and it cannot be changed later.

## Concept: Dimension (Configuration)
**Question:** What is "Dimension" in a vector index?
**Answer:** The fixed length of the vector array (e.g., 1536). It effectively acts as the "Schema" for the index. Every vector stored must have exactly this number of items. It must match the output size of the Embedding Model you are using.

## Concept: Metric (Configuration)
**Question:** What is the "Metric" configuration?
**Answer:** The math formula used to calculate distance/similarity between vectors.
- **Cosine**: Best for text/semantic search (measures direction).
- **Euclidean**: Best for recommendation systems (measures straight-line distance).

## Concept: Pinecone Clients (pc vs index)
**Question:** What is the difference between `pc` and `index` in the Python client?
**Answer:**
- **`pc` (Pinecone Client)**: The "Manager". Used for admin tasks like creating, listing, or deleting indexes.
- **`index` (Index Client)**: The "Worker". Used for data operations like Upserting, Querying, and Fetching within a specific index.

## Concept: Upsert
**Question:** What does "Upsert" do?
**Answer:** A combination of "Update" and "Insert". If a vector with the given ID already exists, it overwrites it. If it does not exist, it creates a new record.

## Concept: Query
**Question:** What is a "Query" operation?
**Answer:** The act of searching the database. You provide a "Query Vector" and ask for the "Top K" (e.g., Top 5) most similar vectors in the database.

## Concept: Namespace
**Question:** What is a Namespace?
**Answer:** A way to rigidly partition data inside a single index (like drawers in a cabinet). A query in one namespace cannot see data in another. It is used for multi-tenancy (e.g., separating Customer A from Customer B).

## Concept: Metadata Filtering
**Question:** What is Metadata Filtering?
**Answer:** A way to refine search results based on tags (e.g., `genre="action"`). Unlike Namespaces (which isolate data completely), Filtering allows you to search the whole database but narrow down results dynamically.

## Code: Correct Upsert to Namespaces
**Question:** How do you upsert items into different namespaces?
**Answer:** You must make separate upsert calls for each namespace. You cannot mix namespaces in a single batch.
```python
# CORRECT
index.upsert(vectors=[("id-1", [0.1, 0.1])], namespace="action")
index.upsert(vectors=[("id-2", [0.2, 0.2])], namespace="comedy")

# INCORRECT (You cannot do this)
# index.upsert([("id-1", ..., "action"), ("id-2", ..., "comedy")])
```

## Code: Querying with Namespaces
**Question:** Why did my query return 0 results even though I upserted data?
**Answer:** You likely upserted into a namespace (e.g., `namespace="action"`) but forgot to specify it in your query. By default, `query()` looks in the empty/default namespace.
```python
# WRONG (Returns 0 matches)
index.query(vector=[...], top_k=5) 

# RIGHT (finds your data)
index.query(vector=[...], top_k=5, namespace="action")
```

## Code: Passing Vectors to Query
**Question:** Can I pass a DataFrame or Index object to `vector=` in `query()`?
**Answer:** No. You must pass a **list of numbers** (floats).
```python
# WRONG
# index.query(vector=df['column'], ...) # Error: Serialization failed

# RIGHT
index.query(vector=[0.1, 0.5, 0.9], ...) 
# OR
index.query(vector=df.iloc[0]['vector_column'], ...)
```

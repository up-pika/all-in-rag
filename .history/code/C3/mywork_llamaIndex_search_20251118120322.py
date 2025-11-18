from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./llamaindex_index_store")
index = load_index_from_storage(storage_context)
```

### 查询索引：
```python
query_engine = index.as_query_engine()
response = query_engine.query("什么是LlamaIndex?")
```

### 更新索引：
```python
# 添加新文档
index.insert(Document(text="新文档内容"))
index.storage_context.persist(persist_dir="./llamaindex_index_store")
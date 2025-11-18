from llama_index.core import Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型（必须与创建索引时使用相同的模型）
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 加载存储的索引
storage_context = StorageContext.from_defaults(persist_dir="./llamaindex_index_store")
index = load_index_from_storage(storage_context)
print("索引加载成功！\n")

# 3. 创建检索器进行相似性搜索
retriever = index.as_retriever(similarity_top_k=3)  # 返回最相似的3个结果

# 4. 执行相似性搜索
query = "什么是LlamaIndex?"
print(f"查询: {query}\n")
print("=" * 60)

# 检索相似文档
nodes = retriever.retrieve(query)

print(f"找到 {len(nodes)} 个相似文档:\n")
for i, node in enumerate(nodes, 1):
    print(f"结果 {i}:")
    print(f"  相似度分数: {node.score:.4f}")
    print(f"  文档内容: {node.text}")
    print(f"  节点ID: {node.node_id}")
    print("-" * 60)

# # 5. 可选：使用查询引擎进行问答（会使用LLM生成答案）
# print("\n" + "=" * 60)
# print("使用查询引擎进行问答:")
# query_engine = index.as_query_engine(similarity_top_k=3)
# response = query_engine.query(query)
# print(f"回答: {response}\n")

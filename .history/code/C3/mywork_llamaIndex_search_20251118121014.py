from llama_index.core import Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json

# 1. 配置全局嵌入模型（必须与创建索引时使用相同的模型）
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 加载存储的索引
storage_context = StorageContext.from_defaults(persist_dir="./llamaindex_index_store")
index = load_index_from_storage(storage_context)
print("索引加载成功！\n")

# 2.1 理解docstore结构：Document vs Node
print("=" * 60)
print("理解 LlamaIndex 存储结构：Document vs Node")
print("=" * 60)
print("\nLlamaIndex 将数据分为两层：")
print("1. Document（文档）：原始输入的文档")
print("2. Node（节点）：从文档转换而来的可索引单元\n")

# 读取docstore.json来展示结构
try:
    with open("./llamaindex_index_store/docstore.json", "r", encoding="utf-8") as f:
        docstore = json.load(f)
    
    metadata = docstore.get("docstore/metadata", {})
    ref_doc_info = docstore.get("docstore/ref_doc_info", {})
    data = docstore.get("docstore/data", {})
    
    # 分离Document和Node
    documents = {}
    nodes = {}
    
    for doc_id, info in metadata.items():
        if "ref_doc_id" in info:
            # 这是Node，有ref_doc_id指向原始Document
            nodes[doc_id] = info
        else:
            # 这是Document
            documents[doc_id] = info
    
    print(f"原始文档数（Document）: {len(documents)}")
    print(f"索引节点数（Node）: {len(nodes)}")
    print(f"总计: {len(metadata)} 个条目\n")
    
    print("文档结构关系：")
    for doc_id, doc_info in ref_doc_info.items():
        node_ids = doc_info.get("node_ids", [])
        print(f"\n文档ID: {doc_id[:8]}...")
        print(f"  └─ 对应的节点ID: {[nid[:8] + '...' for nid in node_ids]}")
        
        # 显示节点内容
        for node_id in node_ids:
            if node_id in data:
                node_data = data[node_id].get("__data__", {})
                text = node_data.get("text", "")
                print(f"     节点内容: {text[:50]}...")
    
    print("\n" + "=" * 60 + "\n")
except Exception as e:
    print(f"无法读取docstore.json: {e}\n")

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

# 5. 可选：使用查询引擎进行问答（会使用LLM生成答案）
print("\n" + "=" * 60)
print("使用查询引擎进行问答:")
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(query)
print(f"回答: {response}\n")

# 6. 更新索引示例（注释掉，避免每次运行都更新）
# 添加新文档
# index.insert(Document(text="新文档内容"))
# index.storage_context.persist(persist_dir="./llamaindex_index_store")

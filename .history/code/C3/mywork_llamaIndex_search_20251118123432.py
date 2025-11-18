import os

# 1. 配置HuggingFace镜像源（必须在导入之前设置，解决SSL网络问题）
# 设置镜像源，将 huggingface.co 替换为 hf-mirror.com
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置本地缓存目录（如果模型已下载，会优先使用本地缓存）
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))

# 禁用 SSL 验证（如果镜像源仍有问题，可以尝试这个，但不推荐用于生产环境）
# os.environ["CURL_CA_BUNDLE"] = ""
# os.environ["REQUESTS_CA_BUNDLE"] = ""

from llama_index.core import Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 2. 配置全局嵌入模型（必须与创建索引时使用相同的模型）
# 如果模型已下载到本地，会自动使用缓存
try:
    # 尝试使用本地缓存或镜像源加载模型
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        trust_remote_code=True
    )
    print("嵌入模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("\n提示：如果网络有问题，可以尝试以下方法：")
    print("1. 检查是否已设置镜像源：HF_ENDPOINT=https://hf-mirror.com")
    print("2. 手动下载模型到本地缓存目录")
    print("3. 使用本地模型路径（如果已下载）")
    print(f"   例如：HuggingFaceEmbedding(model_name='./models/bge-small-zh-v1.5')")
    raise

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

import json
import os

import numpy as np
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
    WeightedRanker,  # 添加 WeightedRanker
    connections,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 初始化设置
COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI = "http://localhost:19530"  # 服务器模式
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相对路径
BATCH_SIZE = 50

# 2. 连接 Milvus 并初始化嵌入模型
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")

# 3. 创建 Collection
milvus_client = MilvusClient(uri=MILVUS_URI)
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(
        name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"]),
]

# 如果集合不存在，则创建它及索引
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")
    # 创建集合
    collection = Collection(
        name=COLLECTION_NAME, schema=schema, consistency_level="Strong"
    )
    print("--> Collection 创建成功。")

    # 4. 创建索引
    print("--> 正在为新集合创建索引...")
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

collection = Collection(COLLECTION_NAME)

# 5. 加载数据并插入
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

if collection.is_empty:
    print(f"--> Collection 为空，开始插入数据...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    docs, metadata = [], []
    for item in dataset:
        parts = [
            item.get("title", ""),
            item.get("description", ""),
            item.get("location", ""),
            item.get("environment", ""),
            # *item.get('combat_details', {}).get('combat_style', []),
            # *item.get('combat_details', {}).get('abilities_used', []),
            # item.get('scene_info', {}).get('time_of_day', '')
        ]
        docs.append(" ".join(filter(None, parts)))
        metadata.append(item)
    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    print("--> 正在生成向量嵌入...")
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")
    # 为每个字段准备批量数据
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["title"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    environments = [doc["environment"] for doc in metadata]

    # 获取向量
    sparse_vectors = embeddings["sparse"]
    dense_vectors = embeddings["dense"]

    # 插入数据
    collection.insert(
        [
            img_ids,
            paths,
            titles,
            descriptions,
            categories,
            locations,
            environments,
            sparse_vectors,
            dense_vectors,
        ]
    )

    collection.flush()
    print(f"--> 数据插入完成，总数: {collection.num_entities}")
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# 6. 执行搜索
search_query = "悬崖上的巨龙"
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
top_k = 5

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

query_embeddings = ef([search_query])
dense_vec = query_embeddings["dense"][0]
sparse_vec = query_embeddings["sparse"]._getrow(0)

# 打印向量信息
print("\n=== 向量信息 ===")
print(f"密集向量维度: {len(dense_vec)}")
print(f"密集向量前5个元素: {dense_vec[:5]}")
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")

print(f"\n稀疏向量维度: {sparse_vec.shape[1]}")
print(f"稀疏向量非零元素数量: {sparse_vec.nnz}")
print("稀疏向量前5个非零元素:")
for i in range(min(5, sparse_vec.nnz)):
    print(f"  - 索引: {sparse_vec.indices[i]}, 值: {sparse_vec.data[i]:.4f}")
density = sparse_vec.nnz / sparse_vec.shape[1] * 100
print(f"\n稀疏向量密度: {density:.8f}%")

# 定义搜索参数
search_params = {"metric_type": "IP", "params": {}}

# 先执行单独的搜索
print("\n--- [单独] 密集向量搜索结果 ---")
dense_results = collection.search(
    [dense_vec],
    anns_field="dense_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,  # 单独搜索需要过滤器
    output_fields=[
        "title",
        "path",
        "description",
        "category",
        "location",
        "environment",
    ],
)[0]

for i, hit in enumerate(dense_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [单独] 稀疏向量搜索结果 ---")
sparse_results = collection.search(
    [sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,  # 单独搜索需要过滤器
    output_fields=[
        "title",
        "path",
        "description",
        "category",
        "location",
        "environment",
    ],
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [混合] RRF 融合搜索结果 ---")
# 创建 RRF 融合器
rerank_rrf = RRFRanker(k=60)

# 创建搜索请求（在每个请求中添加过滤器）
dense_req_rrf = AnnSearchRequest(
    data=[dense_vec],
    anns_field="dense_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
)
sparse_req_rrf = AnnSearchRequest(
    data=[sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
)

# 执行 RRF 混合搜索
results_rrf = collection.hybrid_search(
    [sparse_req_rrf, dense_req_rrf],
    rerank=rerank_rrf,
    limit=top_k,
    output_fields=[
        "title",
        "path",
        "description",
        "category",
        "location",
        "environment",
    ],
)[0]

# 打印 RRF 结果
print("使用 RRF (Reciprocal Rank Fusion) 融合，k=60：")
for i, hit in enumerate(results_rrf):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [混合] 加权线性融合搜索结果 ---")
# 创建加权融合器
# weights: [稀疏向量权重, 密集向量权重]
# 权重和不需要等于1，Milvus会自动归一化
alpha = 0.6  # 密集向量权重
sparse_weight = 1 - alpha  # 0.4 稀疏向量权重
dense_weight = alpha  # 0.6 密集向量权重

rerank_weighted = WeightedRanker(sparse_weight, dense_weight)

# 创建搜索请求
dense_req_weighted = AnnSearchRequest(
    data=[dense_vec],
    anns_field="dense_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
)
sparse_req_weighted = AnnSearchRequest(
    data=[sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
)

# 执行加权混合搜索
# 注意：请求顺序要与权重顺序一致！
results_weighted = collection.hybrid_search(
    [sparse_req_weighted, dense_req_weighted],  # 顺序：稀疏、密集
    rerank=rerank_weighted,
    limit=top_k,
    output_fields=[
        "title",
        "path",
        "description",
        "category",
        "location",
        "environment",
    ],
)[0]

# 打印加权融合结果
print(f"使用加权线性融合，稀疏权重={sparse_weight:.1f}, 密集权重={dense_weight:.1f} (语义优先)：")
for i, hit in enumerate(results_weighted):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# 额外演示：关键词优先的加权融合
print("\n--- [混合] 加权线性融合（关键词优先）搜索结果 ---")
alpha_keyword = 0.3  # 密集向量权重降低
sparse_weight_keyword = 1 - alpha_keyword  # 0.7 稀疏向量权重
dense_weight_keyword = alpha_keyword  # 0.3 密集向量权重

rerank_keyword = WeightedRanker(sparse_weight_keyword, dense_weight_keyword)

dense_req_keyword = AnnSearchRequest(
    data=[dense_vec],
    anns_field="dense_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
)
sparse_req_keyword = AnnSearchRequest(
    data=[sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
)

results_keyword = collection.hybrid_search(
    [sparse_req_keyword, dense_req_keyword],
    rerank=rerank_keyword,
    limit=top_k,
    output_fields=[
        "title",
        "path",
        "description",
        "category",
        "location",
        "environment",
    ],
)[0]

print(f"使用加权线性融合，稀疏权重={sparse_weight_keyword:.1f}, 密集权重={dense_weight_keyword:.1f} (关键词优先)：")
for i, hit in enumerate(results_keyword):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# 融合方法对比分析
print("\n" + "="*60)
print("融合方法对比分析")
print("="*60)
print("\n【不同融合策略的排序对比】")
print(f"查询: '{search_query}'\n")

# 收集所有结果的标题用于对比
def get_titles(results):
    return [hit.entity.get('title') for hit in results]

print("排名  |  RRF融合  |  加权融合(语义)  |  加权融合(关键词)")
print("-" * 70)
max_len = max(len(results_rrf), len(results_weighted), len(results_keyword))
for i in range(max_len):
    rrf_title = get_titles(results_rrf)[i] if i < len(results_rrf) else "-"
    weighted_title = get_titles(results_weighted)[i] if i < len(results_weighted) else "-"
    keyword_title = get_titles(results_keyword)[i] if i < len(results_keyword) else "-"
    print(f"{i+1:2d}    |  {rrf_title:12s}  |  {weighted_title:12s}  |  {keyword_title:12s}")

print("\n【融合方法特点总结】")
print("1. RRF融合 (k=60):")
print("   - 只关注排名，忽略得分差异")
print("   - 鲁棒性强，适合通用场景")
print("   - 无需调参，即插即用")
print("\n2. 加权融合 (语义优先, α=0.6):")
print("   - 密集向量权重60%，稀疏向量权重40%")
print("   - 更依赖语义理解")
print("   - 适合智能问答、推荐系统")
print("\n3. 加权融合 (关键词优先, α=0.3):")
print("   - 密集向量权重30%，稀疏向量权重70%")
print("   - 更依赖关键词匹配")
print("   - 适合电商搜索、型号查询")

# 7. 清理资源
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")

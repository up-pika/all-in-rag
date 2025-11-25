# SelfQueryRetriever 的限制与解决方案

## 问题复现

### 实际情况

```python
# 视频数据
1. BV1Bo4y1A7FU - 时长：390秒  ← 最短
2. BV1ug4y157xA - 时长：1063秒
3. BV1yh411V7ge - 时长：806秒

# 查询："时间最短的视频"
# 期望结果：390秒的视频
# 实际结果：806秒的视频 ❌
```

### 原因分析

```python
# LLM生成的查询
Generated Query: query=' ' filter=None limit=1

# 问题：
1. ❌ 没有排序条件（ORDER BY）
2. ❌ 只有 limit=1（限制返回1条）
3. ❌ 返回的是向量相似度最高的，不是时长最短的
```

## 核心问题

**SelfQueryRetriever 的已知限制**：

| 功能 | 是否支持 | 说明 |
|------|---------|------|
| **过滤（Filter）** | ✅ 支持 | `length > 600`、`author == "张三"` |
| **限制数量（Limit）** | ✅ 支持 | `limit=5` |
| **排序（Order By）** | ❌ **不支持** | `ORDER BY length ASC` |

**原因**：
- LangChain的查询构造器（Query Constructor）不包含排序功能
- 大多数向量数据库（Chroma、Milvus等）的相似度搜索本身就是一种"排序"（按相似度）
- 元数据字段的排序需要额外的逻辑

## 解决方案

### 方案1：客户端排序（推荐）✅

**原理**：获取所有匹配文档，在Python中手动排序

```python
# 检测到排序需求时，自动切换到手动排序
if any(keyword in query for keyword in ["最短", "最长", "最小", "最大"]):
    # 获取所有文档
    all_results = vectorstore.similarity_search("", k=len(all_docs))
    
    # 手动排序
    if "最短" in query:
        all_results = sorted(all_results, 
                           key=lambda x: x.metadata.get("length", float('inf')))
    elif "最长" in query:
        all_results = sorted(all_results, 
                           key=lambda x: x.metadata.get("length", 0), 
                           reverse=True)
    
    # 返回第一个
    results = all_results[:1]
```

**优点**：
- ✅ 简单直接，结果正确
- ✅ 适用于小规模数据（<1000条）
- ✅ 不需要修改底层代码

**缺点**：
- ⚠️  需要加载所有文档到内存
- ⚠️  数据量大时性能差

---

### 方案2：分两步查询 ✅

**原理**：先过滤，再排序

```python
def query_with_sort(query_text, sort_by="length", ascending=True):
    # 步骤1：使用SelfQueryRetriever过滤
    filtered_results = retriever.invoke(query_text)
    
    # 步骤2：手动排序
    sorted_results = sorted(
        filtered_results,
        key=lambda x: x.metadata.get(sort_by, 0),
        reverse=not ascending
    )
    
    return sorted_results

# 使用示例
results = query_with_sort("吴恩达的视频", sort_by="length", ascending=True)
```

**优点**：
- ✅ 利用了SelfQueryRetriever的过滤能力
- ✅ 适合"先筛选再排序"的场景

**缺点**：
- ⚠️  需要两步处理

---

### 方案3：直接查询向量数据库 ✅

**原理**：绕过SelfQueryRetriever，直接操作向量数据库

#### Chroma示例

```python
# 直接使用Chroma的get方法
all_docs = vectorstore.get()

# 转换为Document对象并排序
from langchain.schema import Document

docs = [
    Document(page_content=content, metadata=meta)
    for content, meta in zip(all_docs['documents'], all_docs['metadatas'])
]

# 手动排序
sorted_docs = sorted(docs, key=lambda x: x.metadata.get("length", 0))
shortest = sorted_docs[0]
```

#### Milvus示例

```python
from pymilvus import Collection

collection = Collection("videos")

# 使用expr过滤 + output_fields获取元数据
results = collection.query(
    expr="length > 0",  # 过滤条件
    output_fields=["title", "author", "length"],
    limit=100
)

# Python排序
sorted_results = sorted(results, key=lambda x: x["length"])
shortest = sorted_results[0]
```

**优点**：
- ✅ 完全控制查询逻辑
- ✅ 可以使用数据库的原生功能
- ✅ 性能最优

**缺点**：
- ⚠️  失去了LLM自动解析查询的能力
- ⚠️  需要手动编写过滤条件

---

### 方案4：使用SQL数据库 ✅

**原理**：将元数据存储到支持ORDER BY的数据库

```python
import sqlite3
from langchain_community.vectorstores import Chroma

# 创建SQLite数据库存储元数据
conn = sqlite3.connect('videos.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    title TEXT,
    author TEXT,
    length INTEGER,
    view_count INTEGER
)
''')

# 插入数据
for doc in bili:
    cursor.execute('''
    INSERT OR REPLACE INTO videos VALUES (?, ?, ?, ?, ?)
    ''', (
        doc.metadata['source'],
        doc.metadata['title'],
        doc.metadata['author'],
        doc.metadata['length'],
        doc.metadata['view_count']
    ))
conn.commit()

# 查询最短视频
cursor.execute('''
SELECT * FROM videos 
ORDER BY length ASC 
LIMIT 1
''')
shortest = cursor.fetchone()

# 然后从向量数据库获取完整内容
doc = vectorstore.similarity_search(
    "",
    filter={"source": shortest[0]}
)[0]
```

**优点**：
- ✅ 支持复杂的SQL查询
- ✅ 性能好，适合大规模数据
- ✅ 可以JOIN、GROUP BY等

**缺点**：
- ⚠️  需要维护两个数据源（向量DB + SQL DB）
- ⚠️  架构复杂度增加

---

## 不同查询类型的推荐方案

| 查询类型 | 示例 | 推荐方案 | 原因 |
|---------|------|---------|------|
| **纯过滤** | "时长>600秒的视频" | SelfQueryRetriever | LLM可以正确生成filter |
| **排序** | "时间最短的视频" | 方案1（客户端排序） | 简单有效 |
| **过滤+排序** | "观看超过1万次的最长视频" | 方案2（分两步） | 结合两者优势 |
| **复杂查询** | "2023年发布且时长<10分钟的视频" | 方案4（SQL） | 复杂条件处理 |
| **大规模数据** | >10万条记录 | 方案3（直接查询） | 性能考虑 |

---

## 修改后的完整代码

```python
import logging
import os
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)

# 1. 加载数据（省略...）

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据
metadata_field_info = [
    AttributeInfo(name="title", description="视频标题", type="string"),
    AttributeInfo(name="author", description="视频作者", type="string"),
    AttributeInfo(name="view_count", description="观看次数", type="integer"),
    AttributeInfo(name="length", description="视频时长（秒）", type="integer"),
]

# 4. 创建检索器
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True,
)

# 5. 智能查询函数
def smart_query(query_text):
    """智能查询函数，自动处理排序需求"""
    print(f"\n--- 查询: '{query_text}' ---")
    
    # 检测排序关键词
    sort_keywords = ["最短", "最长", "最小", "最大", "最多", "最少", "最高", "最低"]
    needs_sorting = any(kw in query_text for kw in sort_keywords)
    
    if needs_sorting:
        print("⚠️  检测到排序需求，使用手动排序方案")
        
        # 获取所有文档
        all_results = vectorstore.similarity_search("", k=len(bili))
        
        # 根据查询类型排序
        if "最短" in query_text or "最小" in query_text or "最少" in query_text or "最低" in query_text:
            if "时长" in query_text or "时间" in query_text or "长度" in query_text:
                all_results = sorted(all_results, key=lambda x: x.metadata.get("length", float('inf')))
            elif "观看" in query_text or "播放" in query_text:
                all_results = sorted(all_results, key=lambda x: x.metadata.get("view_count", float('inf')))
        
        elif "最长" in query_text or "最大" in query_text or "最多" in query_text or "最高" in query_text:
            if "时长" in query_text or "时间" in query_text or "长度" in query_text:
                all_results = sorted(all_results, key=lambda x: x.metadata.get("length", 0), reverse=True)
            elif "观看" in query_text or "播放" in query_text:
                all_results = sorted(all_results, key=lambda x: x.metadata.get("view_count", 0), reverse=True)
        
        results = all_results[:1]
        print(f"✅ 排序完成，从 {len(all_results)} 个文档中找到最匹配的")
    
    else:
        print("→ 使用SelfQueryRetriever过滤")
        results = retriever.invoke(query_text)
    
    return results

# 6. 测试查询
test_queries = [
    "时间最短的视频",        # 排序查询
    "时长大于600秒的视频",   # 过滤查询
    "观看次数最多的视频",    # 排序查询
]

for query in test_queries:
    results = smart_query(query)
    
    if results:
        for doc in results:
            print(f"标题: {doc.metadata['title']}")
            print(f"作者: {doc.metadata['author']}")
            print(f"来源: https://www.bilibili.com/video/{doc.metadata['source']}")
            print(f"观看次数: {doc.metadata['view_count']}")
            print(f"时长: {doc.metadata['length']}秒")
            print("=" * 50)
    else:
        print("未找到匹配的视频")
```

---

## 未来改进方向

### LangChain社区的进展

- GitHub Issue: [Support ORDER BY in SelfQueryRetriever](https://github.com/langchain-ai/langchain/issues/...)
- 可能的实现方式：
  ```python
  # 未来可能的API（目前不支持）
  retriever = SelfQueryRetriever.from_llm(
      llm=llm,
      vectorstore=vectorstore,
      metadata_field_info=metadata_field_info,
      enable_sort=True,  # 启用排序功能
  )
  ```

### 自定义QueryConstructor

```python
from langchain.chains.query_constructor.base import QueryConstructor

class SortableQueryConstructor(QueryConstructor):
    """支持排序的查询构造器"""
    
    def construct_query(self, query: str):
        # 解析排序需求
        # ...
        return {
            "filter": ...,
            "sort": {"field": "length", "order": "asc"},
            "limit": 1
        }
```

---

## 总结

### 关键要点

1. **SelfQueryRetriever不支持ORDER BY** - 这是已知限制
2. **过滤（Filter）可以正常工作** - `length > 600` 这类查询没问题
3. **排序需求必须手动处理** - 客户端排序是最简单的方案
4. **小规模数据用方案1，大规模数据用方案3或4**

### 最佳实践

```python
# ✅ 正确做法
if "最短" in query or "最长" in query:
    # 使用手动排序
    results = manual_sort(query)
else:
    # 使用SelfQueryRetriever
    results = retriever.invoke(query)

# ❌ 错误做法
# 期望SelfQueryRetriever自动处理"最短"、"最长"等排序查询
results = retriever.invoke("时间最短的视频")  # 结果会错误！
```

### 问题根源回顾

您遇到的问题：
- 查询："时间最短的视频"
- 期望：390秒的视频
- 实际：806秒的视频

原因：
- SelfQueryRetriever生成了 `limit=1` 但没有排序
- 返回的是相似度最高的，不是时长最短的
- 需要使用手动排序方案才能得到正确结果





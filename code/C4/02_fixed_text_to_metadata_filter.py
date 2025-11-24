import logging
import os

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()

    for doc in docs:
        original = doc.metadata

        # 提取基本元数据字段
        metadata = {
            "title": original.get("title", "未知标题"),
            "author": original.get("owner", {}).get("name", "未知作者"),
            "source": original.get("bvid", "未知ID"),
            "view_count": original.get("stat", {}).get("view", 0),
            "length": original.get("duration", 0),
        }

        doc.metadata = metadata
        bili.append(doc)

except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

print(bili)
# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(name="length", description="视频长度（整数）", type="integer"),
]

# 4. 创建自查询检索器
llm = ChatDeepSeek(
    model="deepseek-chat", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY")
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True,
)

# 5. 执行查询示例
# 注意：SelfQueryRetriever不支持ORDER BY排序
# 对于"最短"、"最长"这类查询，需要使用手动排序
queries = [
    "时间最短的视频",  # 需要排序，会自动切换到手动模式
    "时长大于600秒的视频",  # 过滤查询，SelfQueryRetriever可以正常处理
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    results = retriever.invoke(query)

    # 特殊处理：如果查询涉及"最短"、"最长"、"最多"、"最少"等排序需求
    if any(
        keyword in query for keyword in ["最短", "最长", "最小", "最大", "最多", "最少"]
    ):
        print("⚠️  检测到排序需求，SelfQueryRetriever不支持ORDER BY")
        print("→ 改用手动排序方案...")

        # 获取所有文档
        all_results = vectorstore.similarity_search("", k=len(bili))

        # 根据查询类型排序
        if "最短" in query or "最小" in query or "最少" in query:
            if "时长" in query or "时间" in query or "长度" in query:
                all_results = sorted(
                    all_results, key=lambda x: x.metadata.get("length", float("inf"))
                )
            elif "观看" in query or "播放" in query:
                all_results = sorted(
                    all_results,
                    key=lambda x: x.metadata.get("view_count", float("inf")),
                )
        elif "最长" in query or "最大" in query or "最多" in query:
            if "时长" in query or "时间" in query or "长度" in query:
                all_results = sorted(
                    all_results, key=lambda x: x.metadata.get("length", 0), reverse=True
                )
            elif "观看" in query or "播放" in query:
                all_results = sorted(
                    all_results,
                    key=lambda x: x.metadata.get("view_count", 0),
                    reverse=True,
                )

        # 只取第一个
        results = all_results[:1]
        print(f"✅ 手动排序完成，共找到 {len(all_results)} 个文档")

    if results:
        for doc in results:
            title = doc.metadata.get("title", "未知标题")
            author = doc.metadata.get("author", "未知作者")
            view_count = doc.metadata.get("view_count", "未知")
            length = doc.metadata.get("length", "未知")
            source = doc.metadata.get("source", "未知")
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"来源: https://www.bilibili.com/video/{source}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("=" * 50)
    else:
        print("未找到匹配的视频")

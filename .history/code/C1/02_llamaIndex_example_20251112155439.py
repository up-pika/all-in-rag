import os

# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek

# 在程序启动时加载 .env 文件中的环境变量，例如 API Key 等敏感信息
load_dotenv()

# 配置 LLM（大语言模型）：DeepSeek 用于负责回答问题
# 在 Settings 中设置默认 LLM，后续所有 Index/Query 引擎都会使用该配置
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

# 初始化默认的文本嵌入模型，将文本转换为向量；这里选择 BAAI/bge-small-zh-v1.5
# 该嵌入模型主要针对中文任务，适配后续的语义检索
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 使用 SimpleDirectoryReader 读取指定 Markdown 文件，生成 Document 列表
# 这里直接通过 input_files 指定单个文件路径
docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]
).load_data()

# 基于读取到的文档构建向量索引
# VectorStoreIndex 将文档向量化后存储，便于后续相似度检索
index = VectorStoreIndex.from_documents(docs)

# 将索引转换为查询引擎，提供 .query() 等接口进行问题回答
query_engine = index.as_query_engine()

# 打印当前查询引擎内部使用的 Prompt 模板，帮助了解提示词配置
print(query_engine.get_prompts())

# 发起中文问题查询，并打印回答结果
print(query_engine.query("文中举了哪些例子?"))

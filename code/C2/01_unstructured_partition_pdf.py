from collections import Counter

from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"
# 输出文件路径
output_path = "../../data/C2/pdf/Unstructured_partition_pdf_parse.txt"

# 使用Unstructured加载并解析PDF文档
elements = partition_pdf(filename=pdf_path)

# 准备输出内容
output_lines = []
output_lines.append(
    f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符\n"
)

# 统计元素类型
types = Counter(e.category for e in elements)
output_lines.append(f"元素类型: {dict(types)}\n")

# 显示所有元素
output_lines.append("\n所有元素:\n")
for i, element in enumerate(elements, 1):
    output_lines.append(f"Element {i} ({element.category}):\n")
    output_lines.append(str(element) + "\n")
    output_lines.append("=" * 60 + "\n")

# 将结果写入文件
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(output_lines)

# 同时打印到控制台
print("".join(output_lines))
print(f"\n结果已保存到: {output_path}")

import pandas as pd
import re
from lxml import etree

# --- 配置 ---
INPUT_CSV = "train.csv"          # 原始文件名
OUTPUT_CSV = "train_cleaned.csv" # 清洗后的文件名
# OUTPUT_CSV = "train_cleaned_and_resized.csv" # 清洗后的文件名
COORD_PRECISION = 1              # 坐标保留 1 位小数
REMOVE_EXTRA_ATTRS = True        # 移除冗余属性
ALLOWED_ATTRS = ['d', 'fill', 'stroke', 'stroke-width', 'viewBox', 'xmlns', 'width', 'height']

def path_number_replacer(match):
    """正则替换：将高精度数字转为低精度"""
    num_str = match.group(0)
    try:
        num = float(num_str)
        if num.is_integer():
            return str(int(num))
        return f"{num:.{COORD_PRECISION}f}"
    except:
        return num_str

def clean_single_svg(svg_string):
    """清洗单行 SVG 字符串"""
    if not isinstance(svg_string, str) or not svg_string.strip():
        return svg_string
    
    try:
        # 解析 XML
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(svg_string.encode('utf-8'), parser=parser)
        
        for elem in root.getiterator():
            tag = etree.QName(elem).localname
            
            # 1. 移除冗余属性
            if REMOVE_EXTRA_ATTRS:
                attrs_to_remove = [k for k in elem.attrib if etree.QName(k).localname not in ALLOWED_ATTRS]
                for attr in attrs_to_remove:
                    del elem.attrib[attr]

            # 2. 精简 path 的 d 属性
            if tag == 'path' and 'd' in elem.attrib:
                path_d = elem.attrib['d']
                # 匹配所有浮点数
                elem.attrib['d'] = re.sub(r"[-+]?\d*\.\d+|\d+", path_number_replacer, path_d)

            # 3. 精简 viewBox, width, height
            for attr in ['viewBox', 'width', 'height']:
                if attr in elem.attrib:
                    elem.attrib[attr] = re.sub(r"[-+]?\d*\.\d+|\d+", path_number_replacer, elem.attrib[attr])

        # 转回字符串，不换行，压缩体积
        return etree.tostring(root, encoding='unicode', method='xml').replace('\n', '').strip()
    
    except Exception as e:
        # 如果解析失败（某些 SVG 可能是损坏的），返回原值或空
        return svg_string

# def scale_and_clean_svg(svg_str, target_res=256):
#     try:
#         parser = etree.XMLParser(remove_blank_text=True)
#         root = etree.fromstring(svg_str.encode('utf-8'), parser=parser)
        
#         # 1. 获取原始尺寸
#         viewbox = root.attrib.get('viewBox', '').split()
#         if len(viewbox) == 4:
#             orig_w = float(viewbox[2])
#             orig_h = float(viewbox[3])
#         else:
#             # 如果没 viewBox，尝试找 width/height
#             orig_w = float(root.attrib.get('width', target_res))
#             orig_h = float(root.attrib.get('height', target_res))
        
#         # 2. 计算缩放比例 (保持等比例缩放)
#         scale = target_res / max(orig_w, orig_h)
        
#         # 3. 定义缩放并取整的正则函数
#         def scale_num(match):
#             val = float(match.group(0))
#             new_val = val * scale
#             # 返回 1 位小数
#             return f"{new_val:.1f}".rstrip('0').rstrip('.')

#         # 4. 遍历所有元素缩放坐标属性
#         for elem in root.getiterator():
#             # 缩放所有可能包含坐标的属性
#             for attr in ['d', 'cx', 'cy', 'r', 'x', 'y', 'width', 'height', 'x1', 'y1', 'x2', 'y2']:
#                 if attr in elem.attrib:
#                     val = elem.attrib[attr]
#                     # 正则匹配数字并应用缩放
#                     elem.attrib[attr] = re.sub(r"[-+]?\d*\.\d+|\d+", scale_num, val)
            
#             # 5. 移除多余属性，保持 XML 树紧凑
#             allowed = ['d', 'fill', 'stroke', 'viewBox', 'xmlns', 'cx', 'cy', 'r', 'x', 'y']
#             for a in list(elem.attrib):
#                 if etree.QName(a).localname not in allowed:
#                     del elem.attrib[a]

#         # 6. 重设 viewBox
#         root.attrib['viewBox'] = f"0 0 {target_res} {target_res}"
#         return etree.tostring(root, encoding='unicode').replace('\n', '').strip()
#     except:
#         return None

# --- 执行主逻辑 ---
print(f"正在读取 {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print(f"正在清洗 SVG 数据（共 {len(df)} 行）...")
# 使用 progress_apply (如果安装了 tqdm) 或者直接 apply
df['svg'] = df['svg'].apply(clean_single_svg)
# df['svg'] = df['svg'].apply(scale_and_clean_svg)

print(f"正在保存到 {OUTPUT_CSV}...")
df.to_csv(OUTPUT_CSV, index=False)

print("✅ 清洗完成！")
# 打印对比
print(f"原第一行长度: {len(pd.read_csv(INPUT_CSV)['svg'][0])}")
print(f"新第一行长度: {len(df['svg'][0])}")
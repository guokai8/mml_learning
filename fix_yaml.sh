#!/bin/bash

for file in chapter-*.md; do
  # 检查是否有前置 YAML
  if head -1 "$file" | grep -q "^---"; then
    echo "修复 $file..."
    
    # 提取标题行
    title=$(grep "^title:" "$file" | head -1 | sed 's/title: *//' | tr -d '"')
    
    # 重建文件：移除旧前置，添加新前置
    {
      echo "---"
      echo "title: \"$title\""
      echo "---"
      echo ""
      # 跳过原来的 YAML 部分
      tail -n +$(grep -n "^---$" "$file" | tail -1 | cut -d: -f1 | awk '{print $1+1}') "$file"
    } > "$file.tmp"
    
    mv "$file.tmp" "$file"
  fi
done

echo "完成！"

#!/bin/bash
handler=/home/jhkuang/.conda/envs/PDF/bin/python
config_root=/nas/data/jhkuang/projects/PDFBench_related/PDFBench/configs
# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
export FEISHU_USER="18384629319"

# 加载外部配置
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <config_file>${NC}"
    exit 1
fi

cd /nas/data/jhkuang/projects/PDFBench_related/PDFBench
source "$1"  # 加载 region 配置

for config_path in "${config_paths[@]}"; do
    echo -e "${BLUE}Processing config:${NC} $config_path"
    full_config_path="$config_root/$config_path"
    
    if [ ! -f "$full_config_path" ]; then
        echo -e "${RED}❌ Error: Config file $full_config_path does not exist!${NC}"
        continue
    fi
    
    echo -e "${YELLOW}Starting evaluation...${NC}"
    ${handler} -m src.eval --config "$full_config_path"
    
    if [ $? -eq 0 ]; then
        echo -e "${CYAN} >>>${NC} ${GREEN}Successfully processed: $config_path 🎉${NC}"
    else
        echo -e "${RED} >>>${NC} ${RED}Error occurred while processing: $config_path ❌${NC}"
    fi
    
    echo -e "${BLUE}----------------------------------------${NC}"
done

feishu_msg --msg "${config_paths[*]}" --title "PDFBench-${task_name}"
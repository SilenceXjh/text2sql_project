#!/bin/bash

# 定义基础路径（固定部分）
MODEL_PATH="/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B"
BASE_ADAPTER_PATH="/data0/xjh/text2sql_project/spider-lora-qwen0.5B-r16-mlp/checkpoint-epoch"
BASE_OUTPUT_PATH="/data0/xjh/text2sql_project/experiment_outputs/qwen-coder-0.5B-sft-mlp-"

# 循环遍历 1 到 10
for i in {1..10}; do
    # 拼接当前循环的 adapter_path 和 output_file_path
    ADAPTER_PATH="${BASE_ADAPTER_PATH}${i}/"
    OUTPUT_FILE_PATH="${BASE_OUTPUT_PATH}${i}.json"
    
    # 打印当前执行的信息（方便排查问题）
    echo "========================================"
    echo "开始执行第 ${i} 次循环："
    echo "adapter_path: ${ADAPTER_PATH}"
    echo "output_file_path: ${OUTPUT_FILE_PATH}"
    echo "========================================"
    
    # 执行核心 Python 命令
    python text2sql/generate_sft.py \
        --model_path "${MODEL_PATH}" \
        --adapter_path "${ADAPTER_PATH}" \
        --output_file_path "${OUTPUT_FILE_PATH}"
    
    # 检查命令执行是否成功
    if [ $? -eq 0 ]; then
        echo "第 ${i} 次执行成功！"
    else
        echo "第 ${i} 次执行失败！"
        # 可选：如果某一次失败就退出脚本，注释掉则继续执行后续循环
        # exit 1
    fi
    
    # 可选：每次执行后暂停 1 秒（避免执行过快）
    # sleep 1
done

echo "所有循环执行完毕！"
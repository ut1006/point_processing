#!/bin/bash

# 引数が3つ指定されているか確認
if [ $# -ne 3 ]; then
    echo "Usage: $0 <ply_file> <min_y> <max_y>"
    exit 1
fi

# 引数を取得
PLY_FILE=$1
MIN_Y=$2
MAX_Y=$3

# 1つ目のPythonスクリプトをバックグラウンドで実行
python3 project_section.py "$PLY_FILE" "$MIN_Y" "$MAX_Y" &

# 2つ目のPythonスクリプトをバックグラウンドで実行
python3 slice_ply.py "$PLY_FILE" "$MIN_Y" "$MAX_Y" &

# 全てのバックグラウンドプロセスが終了するのを待つ
wait

echo "Finished."

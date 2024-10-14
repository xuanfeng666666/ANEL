


import os
import itertools

# 创建结果目录
result_dir = 'ICEWS18_REGCN_ANEL_result'
os.makedirs(result_dir, exist_ok=True)

param_grid = {
    'densify_k': [0,3,6,9,12,15,18,21],
    'basic_model_weight': [0.5],
}

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for i, params in enumerate(param_combinations):
    # 根据参数组合生成日志文件名
    log_filename = f"densify_{params['densify_k']}_{params['basic_model_weight']}.log"
    out_log = os.path.join(result_dir, log_filename)

    with open(out_log, 'w') as o_f:
        cmd = (
            f"python main.py -d ICEWS18 --densify_k {params['densify_k']} --basic_model_weight {params['basic_model_weight']} --train-history-len 10 --test-history-len 10 --dilate-len 1 "
            f"--lr 0.001 --n-layers 2 --evaluate-every 1 "
            f"--n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn "
            f"--layer-norm --weight 0.5 --entity-prediction --relation-prediction --add-static-graph "
            f"--angle 10 --discount 1 --task-weight 0.7 --gpu 0 --n-epochs 20"
        )

        print(f"Running command: {cmd}")
        o_f.write(f"** Running command: {cmd} **\n")
        
        # 执行命令并将输出重定向到日志文件
        os.system(f"{cmd} > {out_log} 2>&1")

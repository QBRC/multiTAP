nohup python -u batch_process_feature.py > logs/NSCLC_group175.log 2>&1 &

# taskset sets the CPU to run based on CPU affinity
# CUDA_VISIBLE_DEVICES=3 nohup taskset -c 1,3,5,7,9,11 python -u batch_process_feature.py > logs/feat_extract_test2.log 2>&1 &

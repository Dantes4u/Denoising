env OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES="0,1,2" python3 -m torch.distributed.launch --nproc_per_node=3 project.py

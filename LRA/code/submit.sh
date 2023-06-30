# CUDA_VISIBLE_DEVICES=5,6 python3 run_tasks.py --model softmax --task retrieval --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=5,6 python3 run_tasks.py --model dct-256-false --task retrieval --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=5,6 python3 run_tasks.py --model dct-256-true --task retrieval --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_tasks.py --model softmax --task text --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=5,6 python3 run_tasks.py --model dct-256-false --task text --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_tasks.py --model dct-256-true --task text --log_dir ../tbLogs/

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_tasks.py --model dct-256-true --task listops --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_tasks.py --model dct-256-false --task listops --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_tasks.py --model dct-512-true --task listops --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_tasks.py --model dct-512-false --task listops --log_dir ../tbLogs/

# CUDA_VISIBLE_DEVICES=1,2,3,4 python3 run_tasks.py --model softmax --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=1,2,3,4 python3 run_tasks.py --model dct-512-true --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=1,2,3,4 python3 run_tasks.py --model dct-512-false --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=1,2,3,4 python3 run_tasks.py --model dct-256-true --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs/
# CUDA_VISIBLE_DEVICES=1,2,3,4 python3 run_tasks.py --model dct-256-false --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs/

# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.5-true --task text --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.1-true --task text --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.2-true --task text --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.8-true --task text --log_dir ../tbLogs_idct/

# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.5-true --task listops --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.1-true --task listops --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.2-true --task listops --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.8-true --task listops --log_dir ../tbLogs_idct/

# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.5-true --task retrieval --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.1-true --task retrieval --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.2-true --task retrieval --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.8-true --task retrieval --log_dir ../tbLogs_idct/

# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.5-true --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.1-true --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.2-true --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.8-true --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_idct/


# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.5-true --task image --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.1-true --task image --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.2-true --task image --log_dir ../tbLogs_idct/
# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model dct-0.8-true --task image --log_dir ../tbLogs_idct/







# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task listops --log_dir ../tbLogs_idct/ --layers 0
# CUDA_VISIBLE_DEVICES=1,3 python run_tasks.py --model dct-0.2 --task listops --log_dir ../tbLogs_hyperTune/ --layers 0
# # CUDA_VISIBLE_DEVICES=0,1 python run_tasks.py --model dct-0.5 --task listops --log_dir ../tbLogs_hyperTune/ --layers 0
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task listops --log_dir ../tbLogs_idct/ --layers "0,3"
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.2 --task listops --log_dir ../tbLogs_idct/ --layers "0,3"
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.5 --task listops --log_dir ../tbLogs_idct/ --layers "0,3"

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task image --log_dir ../tbLogs_idct/ --layers 0
# CUDA_VISIBLE_DEVICES=0,2 python run_tasks.py --model dct-0.2 --task image --log_dir ../tbLogs_hyperTune/ --layers 0
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task image --log_dir ../tbLogs_idct/ --layers 1
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.2 --task image --log_dir ../tbLogs_idct/ --layers 1


# # CUDA_VISIBLE_DEVICES=5 python run_tasks.py --model dct-0.1 --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_idct/ --layers 0
# CUDA_VISIBLE_DEVICES=0,2 python run_tasks.py --model dct-0.2 --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_hyperTune/ --layers 0
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_idct/ --layers "0,2"
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.2 --task pathfinder32-curv_contour_length_14 --log_dir ../tbLogs_idct/ --layers "0,2"


# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task retrieval --log_dir ../tbLogs_idct/ --layers 0
# CUDA_VISIBLE_DEVICES=0,2 python run_tasks.py --model dct-0.2 --task retrieval --log_dir ../tbLogs_hyperTune/ --layers 0
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task retrieval --log_dir ../tbLogs_idct/ --layers "0,2"
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.2 --task retrieval --log_dir ../tbLogs_idct/ --layers "0,2"

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task text --log_dir ../tbLogs_idct/ --layers 0
# CUDA_VISIBLE_DEVICES=7 python run_tasks.py --model dct-0.2 --task text --log_dir ../tbLogs_hyperTune/ --layers 0
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.1 --task text --log_dir ../tbLogs_idct/ --layers "0,3"
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_tasks.py --model dct-0.2 --task text --log_dir ../tbLogs_idct/ --layers "0,3"

CUDA_VISIBLE_DEVICES=3 python run_tasks.py --model dct-0.2 --task text --log_dir ../tbLogs_hyperTune/ --layers 0
CUDA_VISIBLE_DEVICES=3 python run_tasks.py --model softmax --task text --log_dir ../tbLogs_hyperTune/ --layers 0
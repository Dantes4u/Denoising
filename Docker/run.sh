docker run \
--name=naletov_d -u $(id -u):$(id -g) \
--shm-size 32G \
--gpus all \
--log-driver=none \
--volume=/home/ilya.naletov/:/workspace/all_projects \
--volume=/home/ilya.naletov/Denoise/Code/Data:/workspace/data \
-it \
--entrypoint \
/bin/bash uniform


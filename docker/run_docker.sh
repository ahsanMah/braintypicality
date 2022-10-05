#!/bin/bash

# --user $(id -u):$(id -g) \
# --user root \
#--mount type=bind,src="/AJAX_STOR/amahmood/",target=/ajax \

	# -d \
	# --mount type=bind,src="/AJAX_STOR/amahmood/docker_tmp/",target=/tmp \
# rm -rf /AJAX_STOR/amahmood/docker_tmp/*
# -p 9999:8888 \

docker run \
	-d \
	--rm \
	--init \
	--name vscode-pytorch \
	--ipc=host \
	-e JUPYTER_ENABLE_LAB=yes \
	--gpus device=all \
	--entrypoint="" \
	--mount type=bind,src=/ASD/ahsan_projects/,target=/ahsan_projects \
	--mount type=bind,src="/BEE/Connectome/ABCD/",target=/DATA \
	--mount type=bind,src="/ASD/ahsan_projects/tensorflow_datasets",target=/root/tensorflow_datasets \
	--mount type=bind,src="/ASD/",target=/ASD \
	ahsanmah/pytorch_sde:latest \
	zsh -c '
	jupyter lab --ip 0.0.0.0 --notebook-dir=/ --no-browser --allow-root
	'
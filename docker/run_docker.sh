#!/bin/bash

docker run \
	-d \
	--rm \
	--init \
	--name texas-docker \
	--ipc=host \
	-e JUPYTER_ENABLE_LAB=yes \
	--gpus device=all \
	--mount type=bind,src=/opt/amahmood/,target=/workspace/ \
	--mount type=bind,src=/ASD/ahsan_projects/braintypicality/workdir/,target=/workdir/ \
	--mount type=bind,src="/BEE/Connectome/ABCD/",target=/DATA \
	--mount type=bind,src="/ASD/ahsan_projects/tensorflow_datasets",target=/root/tensorflow_datasets \
	--mount type=bind,src="/ASD/",target=/ASD \
	--mount type=bind,src="/UTexas",target=/UTexas \
	--mount type=bind,src="/AJAX_STOR/amahmood/docker_tmp/",target=/tmp \
	-p 9999:8888 \
	ahsanmah/pytorch_sde:latest \
	jupyter lab --ip 0.0.0.0 --notebook-dir=/ --no-browser --allow-root

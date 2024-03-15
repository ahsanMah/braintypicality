PORT=5006
docker build . -t vizapp
docker run \
    --rm \
    --mount type=bind,src=/ASD/ahsan_projects/braintypicality/,target=/ASD/ahsan_projects/braintypicality/ \
	--mount type=bind,src="/BEE/Connectome/ABCD/",target=/DATA \
    --mount type=bind,src="/ASD/ahsan_projects/braintypicality/msma-explore",target=/app \
    -p $PORT:$PORT \
    vizapp panel serve /app/heatmap.py --address 0.0.0.0 --global-loading-spinner \
    --allow-websocket-origin liger-set-apparently.ngrok-free.app \
    --websocket-compression-level 6 --num-threads=6

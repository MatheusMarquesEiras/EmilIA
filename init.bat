docker compose -f ./docker/mult/docker-compose.yml up -d
@REM docker exec -w /root/context -it ollama-1 ollama create func -f Modelfile -q Q8_0
uv run e.py
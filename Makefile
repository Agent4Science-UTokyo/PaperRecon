.PHONY: type-check fmt fmt-check lint lint-check check up down exec

check: type-check fmt-check lint-check

type-check:
	pyright

fmt:
	ruff format .

fmt-check:
	ruff format . --check --diff

lint:
	ruff check . --fix

lint-check:
	ruff check .

DOCKER_COMPOSE := $(shell \
	if command -v docker-compose >/dev/null 2>&1; then \
		echo docker-compose; \
	elif docker compose version >/dev/null 2>&1; then \
		echo "docker compose"; \
	fi 2>/dev/null)

up:
	@[ -n "$(DOCKER_COMPOSE)" ] || { echo "ERROR: docker compose not installed"; exit 1; }
	$(DOCKER_COMPOSE) build
	$(DOCKER_COMPOSE) up -d
down:
	@[ -n "$(DOCKER_COMPOSE)" ] || { echo "ERROR: docker compose not installed"; exit 1; }
	$(DOCKER_COMPOSE) down
exec:
	docker exec --detach-keys="ctrl-\\" -it ai-writing bash

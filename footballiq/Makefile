.PHONY: up down logs migrate makemigration test lint shell-api shell-db

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	docker compose exec api alembic upgrade head

makemigration:
	docker compose exec api alembic revision --autogenerate -m "$(name)"

test:
	docker compose exec api pytest tests/ -v

lint:
	ruff check . && mypy services/

shell-api:
	docker compose exec api bash

shell-db:
	docker compose exec db psql -U postgres -d footballiq

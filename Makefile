DC ?= docker compose
EXEC := $(DC) exec -T web

.DEFAULT_GOAL := help

.PHONY: help bootstrap up up-dev down restart logs ps shell dbshell \
        migrate makemigrations superuser fmt lint type hygiene test test-fast cov \
        ingest-faers ingest-labels ingest-ctgov ingest-pubmed normalize-drugs \
        build-signals build-index record-cassettes \
        eval-retrieval eval-labeledness eval-briefs eval-signals eval-all \
        demo-data clean reset

help:  ## Show available targets
	@$(DC) version
	@python -c "import re,pathlib;[print(f'{m.group(1):<22}{m.group(2)}') for m in re.finditer(r'^([a-z-]+):.*?## (.*)$$', pathlib.Path('Makefile').read_text(), re.M)]"

bootstrap:  ## Fresh clone to running app with demo data
	$(DC) build
	$(DC) up -d postgres redis
	$(DC) run --rm web python manage.py migrate
	$(DC) up -d web worker beat
	$(DC) run --rm web signaldesk demo load

up:  ## Start the stack
	$(DC) up -d

up-dev:  ## Start the stack with the tailwind watcher
	$(DC) --profile dev up -d

down:  ## Stop the stack
	$(DC) down

restart:  ## Restart application services
	$(DC) restart web worker beat

logs:  ## Tail logs
	$(DC) logs -f --tail=200

ps:  ## Show service status
	$(DC) ps

shell:  ## Interactive shell in the web container
	$(DC) exec web bash

dbshell:  ## psql session
	$(DC) exec postgres psql -U signaldesk -d signaldesk

migrate:  ## Apply migrations
	$(EXEC) python manage.py migrate

makemigrations:  ## Create migrations
	$(EXEC) python manage.py makemigrations

superuser:  ## Create a Django superuser
	$(DC) exec web python manage.py createsuperuser

fmt:  ## Format with ruff
	$(EXEC) ruff format src tests scripts

lint:  ## Lint with ruff
	$(EXEC) ruff check src tests scripts

type:  ## Type-check with mypy in strict mode
	$(EXEC) mypy

hygiene:  ## Encoding and commit-metadata checks on tracked files
	python scripts/check_hygiene.py

test:  ## Full test suite with coverage
	$(EXEC) pytest --cov --cov-report=term-missing

test-fast:  ## Unit tests only, parallel
	$(EXEC) pytest -m unit -n auto

cov:  ## Coverage HTML report
	$(EXEC) pytest --cov --cov-report=html

ingest-faers:  ## Ingest FAERS quarters (ARGS="--from 2012Q4 --to 2026Q1")
	$(EXEC) signaldesk ingest faers $(ARGS)

ingest-labels:  ## Ingest openFDA SPL drug labels
	$(EXEC) signaldesk ingest labels $(ARGS)

ingest-ctgov:  ## Ingest ClinicalTrials.gov studies with results
	$(EXEC) signaldesk ingest ctgov $(ARGS)

ingest-pubmed:  ## Ingest the PubMed baseline adverse-effects subset
	$(EXEC) signaldesk ingest pubmed $(ARGS)

normalize-drugs:  ## Map FAERS drug strings to RxNorm ingredients
	$(EXEC) signaldesk normalize drugs $(ARGS)

build-signals:  ## Recompute contingency tables and estimators
	$(EXEC) signaldesk signals build $(ARGS)

build-index:  ## Chunk, embed, and build the sparse and dense indexes
	$(EXEC) signaldesk index build $(ARGS)

record-cassettes:  ## Record model responses for offline CI
	$(DC) exec web signaldesk evals record-cassettes

eval-retrieval:  ## Retrieval suite
	$(EXEC) signaldesk evals run retrieval

eval-labeledness:  ## Labeledness suite
	$(EXEC) signaldesk evals run labeledness

eval-briefs:  ## Evidence brief suite
	$(EXEC) signaldesk evals run briefs

eval-signals:  ## Reference-set validation suite
	$(EXEC) signaldesk evals run signals

eval-all:  ## All evaluation suites
	$(EXEC) signaldesk evals run all

demo-data:  ## Load the committed fixture slice
	$(EXEC) signaldesk demo load

clean:  ## Remove containers, keep volumes
	$(DC) down --remove-orphans

reset:  ## Remove containers and volumes, destroying all data
	$(DC) down -v --remove-orphans

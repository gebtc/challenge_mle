SHELL := bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: help
help:             	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:			## Create a virtual environment
	@echo "Creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "Run 'source .venv/bin/activate' to enable the environment"

.PHONY: install
install:		## Install dependencies
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -r requirements.txt

STRESS_URL = http://127.0.0.1:8000
.PHONY: stress-test
stress-test:
	# Override STRESS_URL with your deployed app URL, e.g.:
	#   make stress-test STRESS_URL=https://flight-delay-api.fly.dev
	mkdir -p reports
	locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H $(STRESS_URL)

.PHONY: fly-deploy
fly-deploy:		## Deploy to Fly.io
	fly deploy

.PHONY: fly-open
fly-open:		## Open deployed app in browser
	fly open

.PHONY: fly-logs
fly-logs:		## Tail deployment logs
	fly logs

.PHONY: model-test
model-test:			## Run tests and coverage
	mkdir -p reports
	cd tests && \
	$(ENV_PREFIX)pytest \
		--cov-config=../.coveragerc \
		--cov-report term \
		--cov-report html:../reports/html \
		--cov-report xml:../reports/coverage.xml \
		--junitxml=../reports/junit.xml \
		--cov=../challenge \
		model/test_model.py

.PHONY: api-test
api-test:			## Run tests and coverage
	mkdir -p reports
	cd tests && \
	$(ENV_PREFIX)pytest \
		--cov-config=../.coveragerc \
		--cov-report term \
		--cov-report html:../reports/html \
		--cov-report xml:../reports/coverage.xml \
		--junitxml=../reports/junit.xml \
		--cov=../challenge \
		api/test_api.py

.PHONY: build
build:			## Build the Python wheel using build (PEP 517)
	pip install build
	python -m build --wheel
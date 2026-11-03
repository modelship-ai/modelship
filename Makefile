VERSION := $(shell grep -m1 '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
MAJOR   := $(shell echo $(VERSION) | cut -d. -f1)
MINOR   := $(shell echo $(VERSION) | cut -d. -f2)
PATCH   := $(shell echo $(VERSION) | cut -d. -f3)

BOOTSTRAP_DIR ?= $(HOME)/.cache/mship-bootstrap

.PHONY: unit-test wheel bootstrap lint lint-fix pins release-patch release-minor release-major _release llama-cpp-bump

unit-test:
	uv run pytest tests/ -v -m "not integration"

wheel:
	rm -rf dist
	uv build -o dist .
	uv build -o dist bootstrap

bootstrap:
	@case "$(VARIANT)" in cuda|cpu|metal) ;; *) echo "Error: usage: make bootstrap VARIANT=cuda|cpu|metal" >&2; exit 1;; esac
	@$(MAKE) wheel
	uv venv --allow-existing --python 3.12.10 $(BOOTSTRAP_DIR)/venv
	uv pip install --python $(BOOTSTRAP_DIR)/venv/bin/python --reinstall --no-cache dist/mship-*.whl
	@if [ -n "$$MSHIP_LLAMA_SERVER_BIN" ]; then echo "Ignoring MSHIP_LLAMA_SERVER_BIN=$$MSHIP_LLAMA_SERVER_BIN: mship bootstrap provisions llama-server"; fi
	cd $(BOOTSTRAP_DIR) && env -u MSHIP_LLAMA_SERVER_BIN MSHIP_HOME=$(BOOTSTRAP_DIR)/home UV_FIND_LINKS=$(CURDIR)/dist \
		venv/bin/mship bootstrap --$(VARIANT)
	uv pip install --python $(BOOTSTRAP_DIR)/home/envs/$(VARIANT)/.venv/bin/python --no-deps --reinstall --no-cache \
		dist/mship_engine-*.whl
	@site=$$($(BOOTSTRAP_DIR)/home/envs/$(VARIANT)/.venv/bin/python -I -c \
		"import modelship, os; print(os.path.dirname(os.path.dirname(modelship.__file__)))"); \
	git ls-files modelship | while read -r f; do \
		cmp -s "$$f" "$$site/$$f" || { echo "Error: $$f differs from the installed mship-engine" >&2; exit 1; }; \
	done

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

# Fires all four platform builds for a new llama.cpp tag and returns immediately
# — it does not wait. The workflow publishes one release holding every platform's
# binary, gated on all four succeeding. Updating the llama_cpp.py/Dockerfile pins
# to the new tag is still manual.
llama-cpp-bump:
	@if [ -z "$(TAG)" ]; then echo "Error: usage: make llama-cpp-bump TAG=b10375" >&2; exit 1; fi
	@gh workflow run llama-cpp-build.yml -f tag=$(TAG)
	@echo "Triggered llama.cpp build for $(TAG) (~2h, gated on the CUDA job)."
	@echo "Publishes https://github.com/modelship-ai/llama-cpp-builds/releases/tag/llamacpp-$(TAG)"

release-patch:
	$(eval NEW_VERSION := $(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1))))
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

release-minor:
	$(eval NEW_VERSION := $(MAJOR).$(shell echo $$(($(MINOR)+1))).0)
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

release-major:
	$(eval NEW_VERSION := $(shell echo $$(($(MAJOR)+1))).0.0)
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

_release:
	@if [ "$$(git branch --show-current)" != "main" ]; then echo "Error: releases must be made from the main branch" >&2; exit 1; fi
	@if [ -n "$$(git status --porcelain)" ]; then echo "Error: working tree is dirty, commit or stash changes first" >&2; exit 1; fi
	@git pull --rebase origin main
	@echo "Bumping version: $(VERSION) -> $(NEW_VERSION)"
	@sed -i '0,/^version = ".*"/{s/^version = ".*"/version = "$(NEW_VERSION)"/}' pyproject.toml
	@sed -i '0,/^version = ".*"/{s/^version = ".*"/version = "$(NEW_VERSION)"/}' bootstrap/pyproject.toml
	@sed -i 's/^__version__ = ".*"/__version__ = "$(NEW_VERSION)"/' bootstrap/mship_bootstrap/__init__.py
	@uv lock
	@$(MAKE) pins
	@# --- lockstep the Helm chart with the app version (single source of truth: the tag) ---
	@# chart version == appVersion == image tag == app version, so a checked-out tag
	@# renders an installable chart and `helm install --version X.Y.Z` pairs image X.Y.Z.
	@sed -i 's/^version: .*/version: $(NEW_VERSION)/' helm/modelship/Chart.yaml
	@sed -i 's/^appVersion: .*/appVersion: "$(NEW_VERSION)"/' helm/modelship/Chart.yaml
	@sed -i '0,/^  tag: ".*"/{s/^  tag: ".*"/  tag: "$(NEW_VERSION)"/}' helm/modelship/values.yaml
	@# --- sync monitoring assets into the chart (Helm .Files can't read docs/) ---
	@cp docs/grafana-dashboard.json helm/modelship/files/grafana-dashboard.json
	@cp docs/prometheus-alerts.yml helm/modelship/files/prometheus-alerts.yml
	@# --- auto-update CHANGELOG.md ---
	@PREV_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo ""); \
	if [ -n "$$PREV_TAG" ]; then \
		RANGE="$$PREV_TAG..HEAD"; \
	else \
		RANGE=""; \
	fi; \
	ADDED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^feat(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	FIXED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^fix(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	CHANGED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^(refactor|perf|docs|chore|build|ci|style|test)(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	TMPF=$$(mktemp); \
	echo "" >> "$$TMPF"; \
	echo "## [$(NEW_VERSION)] - $$(date +%Y-%m-%d)" >> "$$TMPF"; \
	if [ -n "$$ADDED" ]; then echo "" >> "$$TMPF"; echo "### Added" >> "$$TMPF"; echo "$$ADDED" >> "$$TMPF"; fi; \
	if [ -n "$$FIXED" ]; then echo "" >> "$$TMPF"; echo "### Fixed" >> "$$TMPF"; echo "$$FIXED" >> "$$TMPF"; fi; \
	if [ -n "$$CHANGED" ]; then echo "" >> "$$TMPF"; echo "### Changed" >> "$$TMPF"; echo "$$CHANGED" >> "$$TMPF"; fi; \
	sed -i "/^The format is based on/r $$TMPF" CHANGELOG.md; \
	rm -f "$$TMPF"
	@git add pyproject.toml uv.lock CHANGELOG.md helm/modelship/Chart.yaml helm/modelship/values.yaml helm/modelship/files \
		bootstrap/pyproject.toml bootstrap/mship_bootstrap/__init__.py bootstrap/mship_bootstrap/pins
	@git commit -m "release: v$(NEW_VERSION)"
	@git tag -a "v$(NEW_VERSION)" -m "Release v$(NEW_VERSION)"
	@git push origin main --follow-tags
	@echo "Done. GitHub Actions will build and publish the release."

# Per-variant hash-pinned dependency lists, shipped in the bootstrapper wheel.
# --locked, not --frozen: a stale lock silently exports the wrong package set.
pins:
	@uv export --quiet --locked --no-emit-project --format requirements-txt \
		-o bootstrap/mship_bootstrap/pins/thin.txt
	@uv export --quiet --locked --no-emit-project --extra cpu --extra vllm-cpu --format requirements-txt \
		-o bootstrap/mship_bootstrap/pins/cpu.txt
	@uv export --quiet --locked --no-emit-project --extra cuda --format requirements-txt \
		-o bootstrap/mship_bootstrap/pins/cuda.txt
	@uv export --quiet --locked --no-emit-project --extra metal --format requirements-txt \
		-o bootstrap/mship_bootstrap/pins/metal.txt

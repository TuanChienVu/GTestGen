# Third-party notices

This repository and its release assets include, or are built from, the third-party material listed below. Each item
keeps its own licence. Nothing here relicenses third-party material.

## 1. Source projects of the dataset

The dataset is mined from the git history of six open-source projects. Files and excerpts from these projects appear
in the records (A1), prompts (A2) and model outputs (A3). The pinned commits are those used to build the dataset;
`data/provenance.jsonl` gives the full commit hash, path, SHA-256 hash, licence file and, where the project has
one, the NOTICE file of each file used: feature files, step definition files, and the files that define suites,
scopes and contexts.

| Repository | Licence (SPDX) | Licence file | Pinned commit | Language |
|---|---|---|---|---|
| [Sylius/Sylius](https://github.com/Sylius/Sylius) | MIT | `LICENSE` | `c096ca3fc015` | PHP |
| [PrestaShop/PrestaShop](https://github.com/PrestaShop/PrestaShop) | OSL-3.0 | `LICENSE.md` | `1f83d7b44599` | PHP |
| [apache/fineract](https://github.com/apache/fineract) | Apache-2.0 | `LICENSE_SOURCE` | `95d444751144` | Java |
| [airslie/renalware](https://github.com/airslie/renalware) | MIT | `MIT-LICENSE` | `7105621abe1c` | Ruby |
| [nhsconnect/gpconnect-provider-testing](https://github.com/nhsconnect/gpconnect-provider-testing) | Apache-2.0 | `LICENSE` | `c59258277e7f` | C# |
| [openMF/ph-ee-integration-test](https://github.com/openMF/ph-ee-integration-test) | MPL-2.0 | `LICENSE` | `4a833d5e36fe` | Java |

Notes:

- PrestaShop states in `LICENSE.md` that the core is under OSL-3.0 and the modules under AFL-3.0. Every PrestaShop
  file used here comes from the core test tree (`tests/`), so OSL-3.0 applies.
- OSL-3.0 and MPL-2.0 have file-level copyleft terms. Excerpts of files under these licences remain under them, and
  their provenance is recorded so that the original files can be found.
- Of the two Apache-2.0 projects, apache/fineract ships a NOTICE file for its source code (`NOTICE_SOURCE`) at the
  pinned commit; nhsconnect/gpconnect-provider-testing has none. apache/fineract also ships `LICENSE_RELEASE` and
  `NOTICE_RELEASE` for its binary releases; the excerpts in this work are source code, so the `_SOURCE` files apply.
  The text of `NOTICE_SOURCE` at commit `95d4447511444a5379d5f666923234a6b031c131`:

  ```text
  Apache Fineract
  Copyright 2008-2025 The Apache Software Foundation

  This product includes software developed at
  The Apache Software Foundation (http://www.apache.org/).
  ```

The licence and notice files at the pinned commits:

| Repository | File | Bytes | SHA-256 |
|---|---|---|---|
| Sylius/Sylius | `LICENSE` | 1069 | `46923e16ba0cb2e76d9de1b9abfa164eb3d16b47e1bf8938bbd9257f076c79f2` |
| Sylius/Sylius | `LICENSE_OF_TRADEMARK_AND_LOGO` | 8906 | `a796e958dc22ddf23ff59e6d529a91d0165d1b636f18c8d9d7edae4c4776d35c` |
| PrestaShop/PrestaShop | `LICENSE.md` | 383 | `7d1c5e4d880b977d3549bee81abef00a782d03ea3f4a7ef49c37eecd32e68bd8` |
| apache/fineract | `LICENSE_SOURCE` | 12315 | `7e3b13f42d05b1da02165c5b0e200c9496a3e22f5642afa6b6909f3b1298cc98` |
| apache/fineract | `NOTICE_SOURCE` | 169 | `3e9b097df85708ed2f323ec56c2692e2dcce686a5e30969c5c096caae6fd05ca` |
| airslie/renalware | `MIT-LICENSE` | 1055 | `71d44299f22e5fb40bf9cd648e78834703be2076f325aaedc7f6f42857ba5dbb` |
| nhsconnect/gpconnect-provider-testing | `LICENSE` | 11357 | `b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1` |
| openMF/ph-ee-integration-test | `LICENSE` | 16726 | `3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04` |

## 2. Sylius in the execution harness

The execution harness (`er/`) runs Behat on Sylius at commit `c096ca3fc01545ee25078ba3df914bd2069c7fa6`.

- No file of this repository or of the release assets contains Sylius source code. A5 holds identifiers only:
  record ids, feature paths, scenario names, file paths and SHA-256 hashes in the overlay manifests, and HTTP routes in
  the traces.
- The overlay trees, which contain Sylius files with the generated step definitions inserted, are not published.
  `scripts/write_overlays.py` rebuilds them from a local clone.
- The Docker build fetches Sylius at the pinned commit, so the images do not redistribute it either.
- Sylius is licensed under the MIT licence, Copyright (c) 2011-present Sylius Sp. z o.o. (`LICENSE` at the pinned
  commit). The Sylius name and logo are not used; see `LICENSE_OF_TRADEMARK_AND_LOGO` in the Sylius repository.
- The files under `er/docker/trace/` are code of this project (MIT), not taken from Sylius or Behat.

## 3. PHP packages (`er/docker/lock/composer.lock`)

Sylius does not commit a Composer lock file, so the lock file used to build the harness image was generated once and
pinned by its SHA-256 hash (`er/docker/lock/SHA256SUMS`). The file is package metadata from Packagist. The packages
themselves are downloaded when the image is built and are not redistributed here. The licences declared in the lock
file are (261 packages in total):

| Section | Packages | Licences declared |
|---|---|---|
| `packages` | 174 | MIT 167, BSD-3-Clause 3, MPL-2.0 2, Artistic-1.0 1, ISC 1 |
| `packages-dev` | 87 | MIT 59, BSD-3-Clause 25, Apache-2.0 2, ISC 1 |

## 4. Container images

- PHP base image `php:8.4-cli`, pinned by digest in `er/docker/Dockerfile`.
- MySQL image `mysql:8.4`, pinned by digest in `er/docker/compose.yaml`.

These images are pulled at build time and are not redistributed.

## 5. Python packages

The Python dependencies are installed from PyPI under their own licences and are not redistributed:

| File | Used for |
|---|---|
| `requirements.txt`, `requirements-phase3.txt` | the CPU code and its tests |
| `requirements-cpu-test.txt` | the CPU tests of the GPU code on an ordinary machine |
| `requirements-gpu.in`, `requirements-gpu.lock` | training and the training canary |
| `requirements-gen.in`, `requirements-gen.lock` | generation |
| `requirements-gen-fallback-{tf,bnb,tf-bnb}.in` | fallback pin sets declared in advance and not used |

Each lock file was frozen by the first bootstrap of its kind and committed before any other machine ran; later
machines install it exactly with `uv pip sync` (`ops/pod_bootstrap.sh`).

## 6. Language models

No model weights are included in this repository or in the release assets, and the trained adapters are not
released. The outputs in A3 were generated with:

| Model | Hugging Face repository | Pinned revision | Licence |
|---|---|---|---|
| CodeLlama-7B-Instruct | `codellama/CodeLlama-7b-Instruct-hf` | `22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed` | Llama 2 Community License Agreement, with its Acceptable Use Policy |
| DeepSeek-Coder-6.7B-Instruct | `deepseek-ai/deepseek-coder-6.7b-instruct` | `e5d64addd26a6a1db0f9b863abf6ee3141936807` | DeepSeek License Agreement, Version 1.0 (23 October 2023) |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | `63a8b081895390a26e140280378bc85ec8bce07a` | Apache License 2.0 |

The licence files at the pinned revisions (`https://huggingface.co/<repository>/blob/<revision>/<file>`):

| Repository | File | SHA-256 |
|---|---|---|
| `codellama/CodeLlama-7b-Instruct-hf` | `LICENSE` | `8c17c2ebb0ea011be9981cc3922db8ca8fa61e828c5d3f44cb6ae342bf80460b` |
| `codellama/CodeLlama-7b-Instruct-hf` | `USE_POLICY.md` | `590d74a5569b8a20eb2a8b0aa869d1d1d3faf6a7fdda1955ae827073c7f502fc` |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | `LICENSE` | `a3a0e9449cb691a12f4de1d03725fd41326614fdeaf5d80b28c51187da0bed0e` |
| `mistralai/Mistral-7B-Instruct-v0.2` | `README.md` (the `license: apache-2.0` field of its model card; the repository has no separate licence file) | `aa431f0f615eae0a34b895020db5ad43ae287b460d1300b5ee225a3f50ff2343` |

The terms of these licences on model outputs are quoted in `LICENSE-DATA.md`.

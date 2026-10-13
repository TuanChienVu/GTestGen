# CPU test image for the GPU code's test suite (tests_gpu) and the pod-parity gate (ops/pod_parity_gate.sh):
#   docker build -t gtestgen-cpu-gate ops/gate_images/cpu-test
#   PODSAFETY_IMAGE=gtestgen-cpu-gate bash ops/pod_parity_gate.sh both
# Linux, no GPU, no network at run time. Versions follow requirements-gpu.lock (pytest, numpy, ...) and
# requirements-phase3.txt (tree-sitter); GPU-only packages (vllm, bitsandbytes) are not installed: the gate runs
# with GATE_ONLY=1 and the tests use a fake vLLM. git and perl (shasum) are on a real pod too.
# CPython 3.12.12 exactly: tools/score_launcher.py, which the scoring runs through, accepts only the interpreter
# version its limits were measured on.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps passwd util-linux coreutils git perl \
 && rm -rf /var/lib/apt/lists/*
# Debian's git 2.39 names a new repository's first branch master; the release tests create bare repositories and
# expect main (with master, three of them fail on a branch that does not exist yet). The pod bootstrap does not run them.
RUN git config --system init.defaultBranch main
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
RUN pip install --no-cache-dir \
      transformers==5.17.0 peft==0.21.0 safetensors==0.8.0 tokenizers==0.23.2 accelerate==1.15.0 \
      huggingface-hub==1.33.0 jinja2==3.1.6 pytest==9.1.1 numpy==2.5.3 pyyaml==6.0.3 \
      sentencepiece==0.2.2 protobuf==6.33.6 defusedxml==0.7.1 \
      tree-sitter==0.26.0 tree-sitter-php==0.24.1 tree-sitter-java==0.23.5 tree-sitter-ruby==0.23.1 \
      tree-sitter-c-sharp==0.23.5
RUN python3 -c "import torch, transformers, peft, pytest; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'pytest', pytest.__version__)" \
 && command -v useradd runuser pgrep install sha256sum stat git shasum

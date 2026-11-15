"""Two loaders sharing one physical GPU via fractional num_gpus:
`frac-share-vllm` (num_gpus=0.6) and `frac-share-llama-server` (num_gpus=0.3),
deployed together so Ray's fractional scheduling packs both onto the same
physical GPU. The cluster fixture pins CUDA_VISIBLE_DEVICES=0 so that packing
is forced rather than hoped for.

`MODEL_CONFIGS`, `_Deployer`, and the `mship_cluster`/`model_deployer`/`client`
fixtures live in conftest.py.
"""

import concurrent.futures
import subprocess

import pytest


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.llama_server
@pytest.mark.cuda
class TestFractionalGpuSharing:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("frac-share-vllm", "frac-share-llama-server")

    def test_both_tenants_are_on_one_physical_gpu(self):
        """The premise the rest of this class rests on. Without it these tests
        pass just as well with the tenants on separate cards."""
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        uuids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        assert len(uuids) == 1, f"expected both tenants on one GPU, compute apps span {len(uuids)}: {uuids}"

    def test_vllm_tenant_serves_requests(self, client):
        completion = client.chat.completions.create(
            model="frac-share-vllm",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
            temperature=0,
        )
        content = completion.choices[0].message.content
        assert content
        assert "paris" in content.lower()

    def test_llama_server_tenant_serves_requests(self, client):
        completion = client.chat.completions.create(
            model="frac-share-llama-server",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
            temperature=0,
        )
        content = completion.choices[0].message.content
        assert content
        assert "paris" in content.lower()

    def test_concurrent_requests_across_both_tenants_succeed(self, client):
        # Both engines already have KV cache/context allocated against their
        # declared share; driving both at once confirms neither starves the other.
        prompts = [
            {
                "model": "frac-share-vllm",
                "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
                "max_tokens": 128,
            },
            {
                "model": "frac-share-llama-server",
                "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
                "max_tokens": 128,
            },
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(client.chat.completions.create, **p) for p in prompts]
            completions = [f.result() for f in futures]

        for completion in completions:
            assert completion.choices[0].message.content
            assert completion.choices[0].finish_reason in ("stop", "length")

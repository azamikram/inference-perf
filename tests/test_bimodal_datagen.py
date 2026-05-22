# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for BimodalDataGenerator."""

from typing import Union
from unittest.mock import MagicMock

import pytest

from inference_perf.apis import CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import (
    APIConfig,
    APIType,
    BimodalConfig,
    DataConfig,
    DataGenType,
    Distribution,
    DistributionType,
)
from inference_perf.datagen.bimodal_datagen import BimodalDataGenerator


def _make_mock_tokenizer(vocab_size: int = 32000) -> MagicMock:
    """Create a mock tokenizer with the expected interface."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.count_tokens.side_effect = lambda text: len(text.split()) if text.strip() else 0
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode.side_effect = lambda tokens, **kwargs: " ".join(str(t) for t in tokens)
    mock_tokenizer.get_tokenizer.return_value = hf_tok
    return mock_tokenizer


def _make_config(
    mode_a_user_prompt_len: Union[int, Distribution] = 10,
    mode_a_output_len: Union[int, Distribution] = 5,
    mode_b_user_prompt_len: Union[int, Distribution] = 100,
    mode_b_output_len: Union[int, Distribution] = 50,
    mode_a_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[APIConfig, DataConfig]:
    api_config = APIConfig(type=APIType.Completion)
    bimodal_config = BimodalConfig(
        mode_a_user_prompt_len=mode_a_user_prompt_len,
        mode_a_output_len=mode_a_output_len,
        mode_b_user_prompt_len=mode_b_user_prompt_len,
        mode_b_output_len=mode_b_output_len,
        mode_a_ratio=mode_a_ratio,
        seed=seed,
    )
    data_config = DataConfig(
        type=DataGenType.Bimodal,
        bimodal=bimodal_config,
    )
    return api_config, data_config


class TestBimodalDataGenerator:
    def test_requires_tokenizer(self) -> None:
        api_config, data_config = _make_config()
        with pytest.raises(ValueError, match="Tokenizer is required"):
            BimodalDataGenerator(api_config, data_config, None)

    def test_requires_bimodal_config(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.Bimodal)
        with pytest.raises(ValueError, match="bimodal config is required"):
            BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_is_mode_a_decision_deterministic(self) -> None:
        api_config, data_config = _make_config(mode_a_ratio=0.5, seed=42)
        gen1 = BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
        gen2 = BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # Decisions should be identical for same seed
        for i in range(100):
            assert gen1._is_mode_a(i) == gen2._is_mode_a(i)

    def test_is_mode_a_decision_different_seeds(self) -> None:
        api_config1, data_config1 = _make_config(mode_a_ratio=0.5, seed=42)
        gen1 = BimodalDataGenerator(api_config1, data_config1, _make_mock_tokenizer())
        api_config2, data_config2 = _make_config(mode_a_ratio=0.5, seed=43)
        gen2 = BimodalDataGenerator(api_config2, data_config2, _make_mock_tokenizer())

        # Decisions should be different for different seeds (at least some)
        decisions1 = [gen1._is_mode_a(i) for i in range(100)]
        decisions2 = [gen2._is_mode_a(i) for i in range(100)]
        assert decisions1 != decisions2

    def test_mode_a_ratio_bounds_0(self) -> None:
        api_config, data_config = _make_config(mode_a_ratio=0.0)
        gen = BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert not any(gen._is_mode_a(i) for i in range(100))

    def test_mode_a_ratio_bounds_1(self) -> None:
        api_config, data_config = _make_config(mode_a_ratio=1.0)
        gen = BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert all(gen._is_mode_a(i) for i in range(100))

    def test_mode_a_ratio_approximate(self) -> None:
        # With ratio 0.3, we should get roughly 30% Mode A requests over a large number of samples
        api_config, data_config = _make_config(mode_a_ratio=0.3, seed=42)
        gen = BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

        num_samples = 1000
        mode_a_count = sum(1 for i in range(num_samples) if gen._is_mode_a(i))
        actual_ratio = mode_a_count / num_samples

        # Allow some tolerance
        assert 0.25 <= actual_ratio <= 0.35

    def test_load_lazy_data_mode_a(self) -> None:
        api_config, data_config = _make_config(
            mode_a_user_prompt_len=10,
            mode_a_output_len=5,
            mode_b_user_prompt_len=100,
            mode_b_output_len=50,
            mode_a_ratio=1.0,  # Always Mode A
        )
        tokenizer = _make_mock_tokenizer()
        gen = BimodalDataGenerator(api_config, data_config, tokenizer)

        lazy = LazyLoadInferenceAPIData(data_index=0)
        result = gen.load_lazy_data(lazy)

        assert isinstance(result, CompletionAPIData)
        assert result.max_tokens == 5
        assert tokenizer.count_tokens(result.prompt) == 10

    def test_load_lazy_data_mode_b(self) -> None:
        api_config, data_config = _make_config(
            mode_a_user_prompt_len=10,
            mode_a_output_len=5,
            mode_b_user_prompt_len=100,
            mode_b_output_len=50,
            mode_a_ratio=0.0,  # Always Mode B
        )
        tokenizer = _make_mock_tokenizer()
        gen = BimodalDataGenerator(api_config, data_config, tokenizer)

        lazy = LazyLoadInferenceAPIData(data_index=0)
        result = gen.load_lazy_data(lazy)

        assert isinstance(result, CompletionAPIData)
        assert result.max_tokens == 50
        assert tokenizer.count_tokens(result.prompt) == 100

    def test_is_mode_a_decision_interleaving(self) -> None:
        api_config, data_config = _make_config(mode_a_ratio=0.5, seed=42)
        gen = BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

        num_samples = 100
        decisions = [gen._is_mode_a(i) for i in range(num_samples)]

        # Count transitions (False -> True or True -> False)
        transitions = 0
        for i in range(1, num_samples):
            if decisions[i] != decisions[i - 1]:
                transitions += 1

        # A non-interleaved sequence (e.g. all Mode A then all Mode B) would have at most 1 transition.
        # An interleaved sequence should have many transitions.
        # For N=100 and ratio=0.5, expected transitions is ~50.
        # We assert that it is at least 20 to ensure good interleaving.
        assert transitions >= 20

    def test_prefix_caching_and_grouping(self) -> None:
        # Configure 2 Mode A groups (prefix len 20) and 3 Mode B groups (prefix len 50)
        api_config, data_config = _make_config(
            mode_a_user_prompt_len=10,
            mode_a_output_len=5,
            mode_b_user_prompt_len=100,
            mode_b_output_len=50,
            mode_a_ratio=0.5,
            seed=42,
        )
        assert data_config.bimodal is not None
        data_config.bimodal.mode_a_system_prompt_len = 20
        data_config.bimodal.mode_a_groups = 2
        data_config.bimodal.mode_b_system_prompt_len = 50
        data_config.bimodal.mode_b_groups = 3

        tokenizer = _make_mock_tokenizer()
        gen = BimodalDataGenerator(api_config, data_config, tokenizer)

        # Generate 50 requests
        num_samples = 50
        results = []
        for i in range(num_samples):
            lazy = LazyLoadInferenceAPIData(data_index=i)
            results.append(gen.load_lazy_data(lazy))

        # Verify prefixes and lengths
        mode_a_prefix_seen = set()
        mode_b_prefix_seen = set()

        for i, res in enumerate(results):
            is_mode_a = gen._is_mode_a(i)
            assert isinstance(res, CompletionAPIData)
            prompt = res.prompt
            words = prompt.split()

            if is_mode_a:
                # Total length: mode_a_system_prompt_len (20) + mode_a_user_prompt_len (10) = 30
                assert tokenizer.count_tokens(prompt) == 30
                # Extract prefix (first 20 tokens/words)
                prefix = " ".join(words[:20])
                mode_a_prefix_seen.add(prefix)
            else:
                # Total length: mode_b_system_prompt_len (50) + mode_b_user_prompt_len (100) = 150
                assert tokenizer.count_tokens(prompt) == 150
                # Extract prefix (first 50 tokens/words)
                prefix = " ".join(words[:50])
                mode_b_prefix_seen.add(prefix)

        # Verify that we saw exactly the configured number of unique prefixes (groups)
        assert len(mode_a_prefix_seen) == 2
        assert len(mode_b_prefix_seen) == 3

        # Verify that requests within the same group share the EXACT same prefix string
        for i, res in enumerate(results):
            is_mode_a = gen._is_mode_a(i)
            assert isinstance(res, CompletionAPIData)
            prompt = res.prompt
            words = prompt.split()
            if is_mode_a:
                group_id = gen._get_group_id(i, 2, "mode_a")
                expected_prefix = gen.mode_a_system_prompts[group_id]
                actual_prefix = " ".join(words[:20])
                assert actual_prefix == expected_prefix
            else:
                group_id = gen._get_group_id(i, 3, "mode_b")
                expected_prefix = gen.mode_b_system_prompts[group_id]
                actual_prefix = " ".join(words[:50])
                assert actual_prefix == expected_prefix

    def test_user_prompt_distribution_sampling(self) -> None:
        # Configure dynamic input and output distributions for Mode A and Mode B requests
        mode_a_input_dist = Distribution(type=DistributionType.NORMAL, mean=50.0, std_dev=10.0, min=30, max=70)
        mode_a_output_dist = Distribution(type=DistributionType.UNIFORM, min=20, max=40)
        mode_b_input_dist = Distribution(type=DistributionType.POISSON, mean=200.0, min=100, max=300)
        mode_b_output_dist = Distribution(type=DistributionType.FIXED, mean=80.0, min=80, max=80)

        api_config, data_config = _make_config(
            mode_a_user_prompt_len=mode_a_input_dist,
            mode_a_output_len=mode_a_output_dist,
            mode_b_user_prompt_len=mode_b_input_dist,
            mode_b_output_len=mode_b_output_dist,
            mode_a_ratio=0.5,
            seed=42,
        )
        tokenizer = _make_mock_tokenizer()
        gen = BimodalDataGenerator(api_config, data_config, tokenizer)

        # Generate 100 requests to test distributions statistically
        num_samples = 100
        mode_a_input_lens = []
        mode_a_output_lens = []
        mode_b_input_lens = []
        mode_b_output_lens = []

        for i in range(num_samples):
            lazy = LazyLoadInferenceAPIData(data_index=i)
            res = gen.load_lazy_data(lazy)
            assert isinstance(res, CompletionAPIData)
            prompt_len = tokenizer.count_tokens(res.prompt)
            output_len = res.max_tokens

            if gen._is_mode_a(i):
                mode_a_input_lens.append(prompt_len)
                mode_a_output_lens.append(output_len)
            else:
                mode_b_input_lens.append(prompt_len)
                mode_b_output_lens.append(output_len)

        # 1. Verify Mode A input lengths (Normal: mean=50, min=30, max=70)
        assert len(mode_a_input_lens) > 10  # make sure we got enough samples
        assert all(30 <= length <= 70 for length in mode_a_input_lens)
        actual_mode_a_input_mean = sum(mode_a_input_lens) / len(mode_a_input_lens)
        # Check statistical convergence (allow a generous bound due to small sample size)
        assert 40 <= actual_mode_a_input_mean <= 60
        # Check that they actually vary (variance is non-zero)
        assert len(set(mode_a_input_lens)) > 1

        # 2. Verify Mode A output lengths (Uniform: min=20, max=40)
        assert len(mode_a_output_lens) > 10
        assert all(20 <= length <= 40 for length in mode_a_output_lens)
        assert len(set(mode_a_output_lens)) > 1

        # 3. Verify Mode B input lengths (Poisson: mean=200, min=100, max=300)
        assert len(mode_b_input_lens) > 10
        assert all(100 <= length <= 300 for length in mode_b_input_lens)
        actual_mode_b_input_mean = sum(mode_b_input_lens) / len(mode_b_input_lens)
        assert 180 <= actual_mode_b_input_mean <= 220
        assert len(set(mode_b_input_lens)) > 1

        # 4. Verify Mode B output lengths (Fixed: mean=80)
        assert len(mode_b_output_lens) > 10
        assert all(length == 80 for length in mode_b_output_lens)
        assert len(set(mode_b_output_lens)) == 1

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
import hashlib
import logging
from typing import Generator, List, Optional

import numpy as np

from inference_perf.apis import CompletionAPIData, InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import resolve_distribution, sample_from_distribution
from .base import DataGenerator, LazyLoadDataMixin
from .datagen_utils import generate_random_exact_length_text, init_vocab_sampling

logger = logging.getLogger(__name__)


class BimodalDataGenerator(DataGenerator, LazyLoadDataMixin):
    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
    ) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.bimodal is None:
            raise ValueError("bimodal config is required for BimodalDataGenerator")

        self.bimodal_config = config.bimodal

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BimodalDataGenerator")

        self.vocab_size, self.special_token_ids, self.valid_token_ids = init_vocab_sampling(self.tokenizer)

        # We use seed for deterministic hash-based decision, and also for random token generation
        self.seed = self.bimodal_config.seed if self.bimodal_config.seed is not None else 42

        # Standardize all Union[int, Distribution] parameters to Distribution
        self.mode_a_user_prompt_dist = resolve_distribution(self.bimodal_config.mode_a_user_prompt_len)
        self.mode_a_output_dist = resolve_distribution(self.bimodal_config.mode_a_output_len)
        self.mode_b_user_prompt_dist = resolve_distribution(self.bimodal_config.mode_b_user_prompt_len)
        self.mode_b_output_dist = resolve_distribution(self.bimodal_config.mode_b_output_len)

        # Pre-generate Mode A system prompts if configured
        self.mode_a_system_prompts: List[str] = []
        if self.bimodal_config.mode_a_system_prompt_len > 0:
            for i in range(self.bimodal_config.mode_a_groups):
                # Seed for this system prompt, offset to avoid collision with request seeds
                prefix_seed = self.seed + 100_000 + i
                self.mode_a_system_prompts.append(
                    self._generate_exact_length_text(self.bimodal_config.mode_a_system_prompt_len, prefix_seed)
                )

        # Pre-generate Mode B system prompts if configured
        self.mode_b_system_prompts: List[str] = []
        if self.bimodal_config.mode_b_system_prompt_len > 0:
            for i in range(self.bimodal_config.mode_b_groups):
                # Seed for this system prompt, offset to avoid collision with request seeds
                prefix_seed = self.seed + 200_000 + i
                self.mode_b_system_prompts.append(
                    self._generate_exact_length_text(self.bimodal_config.mode_b_system_prompt_len, prefix_seed)
                )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def _is_mode_a(self, index: int) -> bool:
        # Deterministic hash-based decision
        hash_input = f"{self.seed}-{index}".encode("utf-8")
        hash_val = hashlib.md5(hash_input).hexdigest()
        # Convert first 8 chars of hex to int
        int_val = int(hash_val[:8], 16)
        # Scale to [0, 1)
        float_val = int_val / (0xFFFFFFFF + 1)
        return float_val < self.bimodal_config.mode_a_ratio

    def _get_group_id(self, index: int, num_groups: int, salt: str) -> int:
        if num_groups <= 1:
            return 0
        # Salt ensures group decision is independent of _is_mode_a decision
        hash_input = f"{self.seed}-{salt}-{index}".encode("utf-8")
        hash_val = hashlib.md5(hash_input).hexdigest()
        int_val = int(hash_val[:8], 16)
        return int_val % num_groups

    def _generate_exact_length_text(self, target_len: int, request_seed: int) -> str:
        """Generates a string that tokenizes to exactly target_len."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating exact length prompts.")
        # Use a request-specific RNG to ensure independence of request generation order if parallelized,
        # but still deterministic for the same request index.
        request_rng = np.random.default_rng(request_seed)
        return generate_random_exact_length_text(request_rng, self.valid_token_ids, self.tokenizer, target_len)

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        n = data.data_index

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BimodalDataGenerator")

        if self.api_config.type == APIType.Completion:
            # Derived seeds to guarantee independent, deterministic RNG sequences per request index
            sampling_seed = self.seed + n
            text_seed = self.seed + 500_000 + n

            sampling_rng = np.random.default_rng(sampling_seed)

            if self._is_mode_a(n):
                # Sample input and output lengths from resolved distributions
                user_prompt_len = int(sample_from_distribution(self.mode_a_user_prompt_dist, 1, rng=sampling_rng)[0])
                output_len = int(sample_from_distribution(self.mode_a_output_dist, 1, rng=sampling_rng)[0])
                req_type = "mode_a"

                if self.bimodal_config.mode_a_system_prompt_len > 0:
                    group_id = self._get_group_id(n, self.bimodal_config.mode_a_groups, "mode_a")
                    system_prompt = self.mode_a_system_prompts[group_id]
                    unique_text = self._generate_exact_length_text(user_prompt_len, text_seed)
                    text = system_prompt + " " + unique_text
                else:
                    text = self._generate_exact_length_text(user_prompt_len, text_seed)
            else:
                # Sample input and output lengths from resolved distributions
                user_prompt_len = int(sample_from_distribution(self.mode_b_user_prompt_dist, 1, rng=sampling_rng)[0])
                output_len = int(sample_from_distribution(self.mode_b_output_dist, 1, rng=sampling_rng)[0])
                req_type = "mode_b"

                if self.bimodal_config.mode_b_system_prompt_len > 0:
                    group_id = self._get_group_id(n, self.bimodal_config.mode_b_groups, "mode_b")
                    system_prompt = self.mode_b_system_prompts[group_id]
                    unique_text = self._generate_exact_length_text(user_prompt_len, text_seed)
                    text = system_prompt + " " + unique_text
                else:
                    text = self._generate_exact_length_text(user_prompt_len, text_seed)

            logger.debug(f"Request {n} is {req_type} (user_prompt={user_prompt_len}, output={output_len})")
            return CompletionAPIData(prompt=text, max_tokens=output_len)
        else:
            raise Exception("Unsupported API type")

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.api_config.type != APIType.Completion:
            raise Exception(f"Unsupported API type: {self.api_config}. BimodalDataGenerator only supports Completion.")

        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1

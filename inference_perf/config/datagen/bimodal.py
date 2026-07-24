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
from typing import Optional, Union

from pydantic import BaseModel, Field

from inference_perf.config.common import Distribution


class BimodalConfig(BaseModel):
    """Configuration for bimodal data generator.

    Generates a mix of two distinct request distributions (Mode A and Mode B)
    based on configurable lengths, ratios, and grouped KV cache prefixes.
    """

    mode_a_system_prompt_len: int = Field(
        0, ge=0, description="Length of shared system prompt prefix (KV cache) for Mode A requests"
    )
    mode_a_groups: int = Field(1, ge=1, description="Number of KV cache groups for Mode A requests")
    mode_a_user_prompt_len: Union[int, Distribution] = Field(
        10, description="Length or distribution of Mode A user prompt in tokens"
    )
    mode_a_output_len: Union[int, Distribution] = Field(
        10, description="Length or distribution of Mode A output generation in tokens"
    )

    mode_b_system_prompt_len: int = Field(
        0, ge=0, description="Length of shared system prompt prefix (KV cache) for Mode B requests"
    )
    mode_b_groups: int = Field(1, ge=1, description="Number of KV cache groups for Mode B requests")
    mode_b_user_prompt_len: Union[int, Distribution] = Field(
        1024, description="Length or distribution of Mode B user prompt in tokens"
    )
    mode_b_output_len: Union[int, Distribution] = Field(
        1024, description="Length or distribution of Mode B output generation in tokens"
    )

    mode_a_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Proportion of Mode A requests (0.0 to 1.0)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    share_prefix: bool = Field(False, description="Whether Mode A and Mode B should share a prefix")

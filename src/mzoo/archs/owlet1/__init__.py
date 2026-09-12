# Copied from https://github.com/huggingface/transformers/pull/48721/blob/f026c8315a275221a61e087edc3536f341e55f09/src/transformers/models/deepseek_v41/modeling_deepseek_v41.py
# (unmerged PR adding deepseek_v41; text backbone of deepseek-ai/DeepSeek-V4.1-Flash)
# License: Apache-2.0 (HuggingFace). Local modifications (relative -> absolute imports
# so this runs outside the transformers package tree) are marked with 'mzoo:' comments.
# This file was originally auto-generated from modular_deepseek_v41.py; that file was
# not copied here since it is only a codegen source, not needed at runtime.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

# owlet1: a split-into-small-files fork of the frozen dsv4 baseline
# (`src/mzoo/archs/dsv4/`). Class names are intentionally kept as `DeepseekV41*`
# so diffs against the baseline stay clean.

from .cache import DeepseekV41CSACache
from .config import DeepseekV41Config, DeepseekV41TextConfig
from .decoder import DeepseekV41ForCausalLM, DeepseekV41TextModel
from .engram import DeepseekV41EngramEmbedding, DeepseekV41NgramHashState, EngramLayout
from .model import build

__all__ = [
    "build",
    "DeepseekV41Config",
    "DeepseekV41TextConfig",
    "DeepseekV41ForCausalLM",
    "DeepseekV41TextModel",
    "DeepseekV41CSACache",
    "DeepseekV41EngramEmbedding",
    "DeepseekV41NgramHashState",
    "EngramLayout",
]

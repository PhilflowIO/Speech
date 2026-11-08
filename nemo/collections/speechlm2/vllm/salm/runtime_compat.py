# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Model-scoped vLLM adapters for serving SpeechLM.

Prompt contract: audio remains registered through vLLM's multimodal parser while
the rendered user content matches ``SALMDataset``: text is followed by one ASCII
space and one ordered ``<|audio|>`` locator per audio payload, with adjacent
locators separated by one ASCII space.

Marlin padding: vLLM's padding of misaligned MoE intermediates assumes gated
experts; see ``install_marlin_gateless_moe_padding``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from functools import wraps
from typing import Any

from nemo.collections.speechlm2.vllm.salm.config import SPEECHLM_MTP_MODEL_TYPE, TRANSCRIBE_MTP_MODEL_TYPE

_LOGGER = logging.getLogger(__name__)
_INT_WNA16_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int_wna16"

_AUDIO_LOCATOR = "<|audio|>"
_AUDIO_PART_TYPES = frozenset({"audio_url", "input_audio", "audio_embeds"})
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


def _is_nemo_speechlm_tracker(mm_tracker: Any) -> bool:
    """Return whether a vLLM chat tracker belongs to a SpeechLM model."""
    model_config = getattr(mm_tracker, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) in {
        "nemo_speechlm",
        SPEECHLM_MTP_MODEL_TYPE,
        TRANSCRIBE_MTP_MODEL_TYPE,
    }


def _part_type(part: Any) -> str:
    if isinstance(part, str):
        return "text"
    if not isinstance(part, dict):
        raise TypeError(
            "NeMo SpeechLM exact prompt rendering requires string or mapping "
            f"content parts; got {type(part).__name__}."
        )
    part_type = part.get("type")
    if part_type is None:
        direct_audio_types = [kind for kind in _AUDIO_PART_TYPES if kind in part]
        if len(direct_audio_types) == 1:
            return direct_audio_types[0]
    if not isinstance(part_type, str):
        raise TypeError(
            "NeMo SpeechLM exact prompt rendering requires every mapping "
            "content part to have an unambiguous string 'type'."
        )
    return part_type


def _text_value(part: Any, part_type: str) -> str:
    if isinstance(part, str):
        return part
    value = part.get("text")
    if not isinstance(value, str):
        raise TypeError(
            "NeMo SpeechLM exact prompt rendering requires textual content " f"for part type {part_type!r}."
        )
    return value


def _canonicalize_speechlm_parts(parts: Iterable[Any]) -> list[Any]:
    """Retain ordered audio payloads but render text followed by matching locators."""
    parts = list(parts)
    audio_parts: list[Any] = []
    text_values: list[str] = []
    for part in parts:
        part_type = _part_type(part)
        if part_type in _AUDIO_PART_TYPES:
            audio_parts.append(part)
        elif part_type in _TEXT_PART_TYPES:
            text_values.append(_text_value(part, part_type))
        else:
            raise ValueError(
                "NeMo SpeechLM exact prompt rendering only supports text and "
                f"audio content parts; got {part_type!r}."
            )

    if not audio_parts:
        return parts

    text = " ".join(text_values)
    locator_suffix = " ".join([_AUDIO_LOCATOR] * len(audio_parts))
    locator_count = text.count(_AUDIO_LOCATOR)
    if locator_count:
        if locator_count != len(audio_parts):
            raise ValueError(
                f"NeMo SpeechLM prompt contains {locator_count} {_AUDIO_LOCATOR} locators "
                f"for {len(audio_parts)} audio payloads; counts must match."
            )
        expected = (
            locator_suffix
            if text == locator_suffix
            else f"{text.removesuffix(locator_suffix).rstrip()} {locator_suffix}"
        )
        if text != expected:
            raise ValueError(
                "A pre-existing NeMo SpeechLM <|audio|> locator must be final "
                "and be preceded by exactly one ASCII space."
            )
        canonical_text = text
    elif text:
        canonical_text = f"{text} {locator_suffix}"
    else:
        canonical_text = locator_suffix

    # The media item is still parsed and registered. Its explicit locator in
    # the final text consumes the registered placeholder, preventing vLLM from
    # prepending a missing locator.
    return [*audio_parts, {"type": "text", "text": canonical_text}]


_LEGACY_PARSER_NAME = "_parse_chat_message_content_parts"
_PUBLIC_PARSER_NAME = "parse_chat_message_content_parts"


def _patch_speechlm_training_prompt_contract(chat_utils: Any, parser_name: str = _LEGACY_PARSER_NAME) -> None:
    """Install an idempotent, SpeechLM-only exact prompt adapter.

    ``parser_name`` is the attribute of ``chat_utils`` that vLLM's chat path
    resolves the content-part parser through (see ``_locate_chat_parser``).
    """
    original = getattr(chat_utils, parser_name, None)
    if not callable(original):
        raise TypeError(
            f"Unsupported vLLM chat API: {parser_name} is missing. "
            "Update the NeMo SpeechLM compatibility adapter before serving."
        )
    if getattr(original, "_nemo_speechlm_exact_training_prompt", False):
        return

    @wraps(original)
    def _parse_chat_message_content_parts(
        role,
        parts,
        mm_tracker,
        *,
        wrap_dicts,
        interleave_strings,
        mm_processor_kwargs=None,
        multimodal_content_part_separator="\n",
    ):
        if _is_nemo_speechlm_tracker(mm_tracker):
            materialized_parts = list(parts)
            parts = materialized_parts
            has_audio = any(_part_type(part) in _AUDIO_PART_TYPES for part in materialized_parts)
            if has_audio:
                if wrap_dicts or interleave_strings:
                    raise RuntimeError(
                        "NeMo SpeechLM exact prompt rendering requires vLLM's "
                        "non-interleaved string chat-content format."
                    )
                if multimodal_content_part_separator != "\n":
                    raise RuntimeError(
                        "NeMo SpeechLM exact prompt rendering does not accept a "
                        "custom multimodal content-part separator."
                    )
                parts = _canonicalize_speechlm_parts(materialized_parts)
        return original(
            role,
            parts,
            mm_tracker,
            wrap_dicts=wrap_dicts,
            interleave_strings=interleave_strings,
            mm_processor_kwargs=mm_processor_kwargs,
            multimodal_content_part_separator=multimodal_content_part_separator,
        )

    _parse_chat_message_content_parts._nemo_speechlm_exact_training_prompt = True
    setattr(chat_utils, parser_name, _parse_chat_message_content_parts)


def _locate_chat_parser() -> tuple[Any, str]:
    """Return the module and attribute through which vLLM's chat path finds the content-part parser.

    The parser keeps its signature across releases but has moved twice:

    * vLLM 0.33 split ``vllm.renderers.chat_utils`` into a package and made the
      parser public as ``chat_message.parse_chat_message_content_parts``;
      ``parse_chat_message_content`` calls it as a global of ``chat_message``, so
      that module is the one to patch (the package re-export is a separate name).
    * vLLM 0.32 has ``vllm.renderers.chat_utils._parse_chat_message_content_parts``.
    * Earlier releases have ``vllm.entrypoints.chat_utils._parse_chat_message_content_parts``.
    """
    try:
        from vllm.renderers.chat_utils import chat_message  # vLLM >= 0.33
    except ImportError:
        chat_message = None
    if callable(getattr(chat_message, _PUBLIC_PARSER_NAME, None)):
        return chat_message, _PUBLIC_PARSER_NAME

    try:
        from vllm.renderers import chat_utils  # vLLM 0.32
    except ImportError:
        from vllm.entrypoints import chat_utils
    return chat_utils, _LEGACY_PARSER_NAME


def install_prompt_contract() -> None:
    """Install the SpeechLM-only vLLM chat rendering adapter."""
    _patch_speechlm_training_prompt_contract(*_locate_chat_parser())


def _w13_shard_count(columns: int, unit: int) -> int:
    """Return how many ``unit``-wide w13 shards a tensor with ``columns`` columns holds.

    Gated experts store gate and up side by side (2 shards), gate-less experts
    only up (1 shard). Nothing else is legal, so a shape that fits neither is an
    error rather than something to pad.
    """
    if unit > 0 and columns in (unit, 2 * unit):
        return columns // unit
    raise ValueError(
        f"Cannot pad MoE w13 weights: {columns} columns is neither one nor two shards of {unit} columns."
    )


def _pad_w13_shard_cols(x: Any, unit: int, padded_unit: int) -> Any:
    """Zero-pad each w13 shard of an ``(E, rows, shards * unit)`` tensor to ``padded_unit`` columns."""
    if padded_unit == unit:
        return x
    from torch.nn.functional import pad

    experts, rows, columns = x.shape
    shards = _w13_shard_count(columns, unit)
    padded = pad(x.view(experts, rows, shards, unit), (0, padded_unit - unit))
    return padded.reshape(experts, rows, shards * padded_unit).contiguous()


def _pad_w13_bias(bias: Any, n: int, padded_n: int) -> Any:
    """Zero-pad each w13 shard of an ``(E, shards * n)`` bias to ``padded_n``."""
    if padded_n == n:
        return bias
    from torch.nn.functional import pad

    experts, columns = bias.shape
    shards = _w13_shard_count(columns, n)
    padded = pad(bias.view(experts, shards, n), (0, padded_n - n))
    return padded.reshape(experts, shards * padded_n).contiguous()


def install_marlin_gateless_moe_padding() -> None:
    """Make vLLM's Marlin MoE padding handle gate-less experts.

    vLLM 0.33 zero-pads a misaligned intermediate size to a Marlin thread tile, but
    its helpers reshape w13 as two shards. NemotronH's experts are gate-less, so
    their w13 has one shard and the reshape raises at weight loading. The
    replacements read the shard count off the tensor, which leaves gated experts
    untouched. If a vLLM release restructures these helpers there is nothing to
    correct here; a remaining mismatch would still fail loudly at weight loading.
    """
    try:
        int_wna16 = importlib.import_module(_INT_WNA16_MODULE)
    except ImportError:
        _LOGGER.warning("%s is not importable; not patching Marlin MoE padding.", _INT_WNA16_MODULE)
        return
    for name, replacement in (("_pad_w13_shard_cols", _pad_w13_shard_cols), ("_pad_w13_bias", _pad_w13_bias)):
        if callable(getattr(int_wna16, name, None)):
            setattr(int_wna16, name, replacement)
        else:
            _LOGGER.warning("vLLM has no %s.%s; not patching it.", _INT_WNA16_MODULE, name)

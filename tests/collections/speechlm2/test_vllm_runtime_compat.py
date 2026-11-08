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

"""Focused tests for SpeechLM's exact vLLM training-time prompt contract."""

import sys
from types import SimpleNamespace

import pytest
import torch

from nemo.collections.speechlm2.vllm.salm.runtime_compat import (
    _pad_w13_bias,
    _pad_w13_shard_cols,
    _patch_speechlm_training_prompt_contract,
    install_marlin_gateless_moe_padding,
    install_prompt_contract,
)

_AUDIO = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AA=="}}


def _tracker(model_type):
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)))


def _part_type(part):
    if isinstance(part, str):
        return "text"
    return part.get("type") or ("audio_url" if "audio_url" in part else None)


def _fake_vllm_chat_utils(parser_name="_parse_chat_message_content_parts"):
    """Model vLLM 0.23's non-interleaved placeholder assembly."""
    calls = []

    def parse(
        role,
        parts,
        mm_tracker,
        *,
        wrap_dicts,
        interleave_strings,
        mm_processor_kwargs=None,
        multimodal_content_part_separator="\n",
    ):
        parts = list(parts)
        audio_parts = [part for part in parts if _part_type(part) == "audio_url"]
        texts = [part if isinstance(part, str) else part["text"] for part in parts if _part_type(part) == "text"]
        text_prompt = "\n".join(texts)
        missing = len(audio_parts) - text_prompt.count("<|audio|>")
        if missing < 0:
            raise ValueError("more placeholders than audio payloads")
        pieces = ["<|audio|>"] * missing
        if text_prompt:
            pieces.append(text_prompt)
        rendered = multimodal_content_part_separator.join(pieces)
        calls.append(
            {
                "parts": parts,
                "registered_audio_parts": audio_parts,
                "rendered": rendered,
                "separator": multimodal_content_part_separator,
            }
        )
        return [{"role": role, "content": rendered}]

    return SimpleNamespace(**{parser_name: parse}), calls


def _parse(chat_utils, parts, model_type="nemo_speechlm", **kwargs):
    return chat_utils._parse_chat_message_content_parts(
        "user",
        parts,
        _tracker(model_type),
        wrap_dicts=kwargs.pop("wrap_dicts", False),
        interleave_strings=kwargs.pop("interleave_strings", False),
        **kwargs,
    )


def test_audio_only_keeps_payload_and_renders_one_locator():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    result = _parse(chat_utils, [_AUDIO])

    assert result == [{"role": "user", "content": "<|audio|>"}]
    assert calls[0]["registered_audio_parts"] == [_AUDIO]
    assert calls[0]["parts"] == [_AUDIO, {"type": "text", "text": "<|audio|>"}]


def test_text_then_one_ascii_space_then_audio_locator_exactly():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    result = _parse(
        chat_utils,
        [_AUDIO, {"type": "text", "text": "Use these names:"}, " Alice, Bob"],
    )

    assert result == [{"role": "user", "content": "Use these names:  Alice, Bob <|audio|>"}]
    assert calls[0]["registered_audio_parts"] == [_AUDIO]
    assert calls[0]["rendered"].endswith(" <|audio|>")
    assert "\n" not in calls[0]["rendered"]


@pytest.mark.parametrize("model_type", ["nemo_speechlm", "nemo_speechlm_mtp", "nemotron3_5_transcribe_mtp"])
def test_contract_covers_every_speechlm_model_type(model_type):
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    result = _parse(chat_utils, [_AUDIO, {"type": "text", "text": "Transcribe"}], model_type=model_type)

    assert result == [{"role": "user", "content": "Transcribe <|audio|>"}]


def test_text_only_generator_is_preserved():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)
    parts = (part for part in [{"type": "text", "text": "Hello"}])

    result = _parse(chat_utils, parts)

    assert result == [{"role": "user", "content": "Hello"}]
    assert calls[0]["parts"] == [{"type": "text", "text": "Hello"}]
    assert calls[0]["registered_audio_parts"] == []


def test_non_speechlm_is_bit_for_bit_unmodified():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)
    parts = [_AUDIO, {"type": "text", "text": "Transcribe"}]

    result = _parse(chat_utils, parts, model_type="other_multimodal")

    assert result == [{"role": "user", "content": "<|audio|>\nTranscribe"}]
    assert calls[0]["parts"] == parts
    assert calls[0]["separator"] == "\n"


def test_valid_preexisting_locator_is_preserved_without_duplication():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    result = _parse(chat_utils, [_AUDIO, {"type": "text", "text": "Transcribe <|audio|>"}])

    assert result == [{"role": "user", "content": "Transcribe <|audio|>"}]
    assert calls[0]["rendered"].count("<|audio|>") == 1
    assert calls[0]["registered_audio_parts"] == [_AUDIO]


@pytest.mark.parametrize(
    "text",
    [
        "<|audio|> Transcribe",
        "Transcribe  <|audio|>",
        "Transcribe <|audio|> extra",
        "Transcribe <|audio|> <|audio|>",
    ],
)
def test_malformed_preexisting_locator_fails_closed(text):
    chat_utils, _ = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    with pytest.raises(ValueError, match="locator"):
        _parse(chat_utils, [_AUDIO, {"type": "text", "text": text}])


def test_multiple_audio_payloads_are_registered_in_order_with_matching_locators():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    audio_2 = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AQ=="}}
    result = _parse(chat_utils, [_AUDIO, audio_2, {"type": "text", "text": "Compare"}])

    assert result == [{"role": "user", "content": "Compare <|audio|> <|audio|>"}]
    assert calls[0]["registered_audio_parts"] == [_AUDIO, audio_2]
    assert calls[0]["rendered"].count("<|audio|>") == 2


def test_multiple_preexisting_locators_must_match_payload_count_and_exact_suffix():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)
    audio_2 = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AQ=="}}
    audio_3 = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,Ag=="}}

    result = _parse(
        chat_utils,
        [
            _AUDIO,
            audio_2,
            audio_3,
            {"type": "text", "text": "Compare <|audio|> <|audio|> <|audio|>"},
        ],
    )
    assert result == [{"role": "user", "content": "Compare <|audio|> <|audio|> <|audio|>"}]
    assert calls[0]["registered_audio_parts"] == [_AUDIO, audio_2, audio_3]

    with pytest.raises(ValueError, match="counts must match"):
        _parse(chat_utils, [_AUDIO, audio_2, {"type": "text", "text": "Compare <|audio|>"}])
    with pytest.raises(ValueError, match="final"):
        _parse(
            chat_utils,
            [_AUDIO, audio_2, {"type": "text", "text": "<|audio|> <|audio|> Compare"}],
        )


def test_noninterleaved_string_contract_is_enforced():
    chat_utils, _ = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    with pytest.raises(RuntimeError, match="non-interleaved string"):
        _parse(chat_utils, [_AUDIO], wrap_dicts=True)
    with pytest.raises(RuntimeError, match="non-interleaved string"):
        _parse(chat_utils, [_AUDIO], interleave_strings=True)


def test_adapter_is_idempotent_and_fails_closed_on_vllm_api_change():
    chat_utils, _ = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)
    patched = chat_utils._parse_chat_message_content_parts

    _patch_speechlm_training_prompt_contract(chat_utils)

    assert chat_utils._parse_chat_message_content_parts is patched
    with pytest.raises(TypeError, match="_parse_chat_message_content_parts"):
        _patch_speechlm_training_prompt_contract(SimpleNamespace())


def test_install_prompt_contract_patches_vllm_renderers_chat_utils(monkeypatch):
    """vLLM 0.32 moved the chat parser to ``vllm.renderers.chat_utils``."""
    chat_utils, _ = _fake_vllm_chat_utils()
    monkeypatch.setitem(sys.modules, "vllm.renderers", SimpleNamespace(chat_utils=chat_utils))

    install_prompt_contract()

    assert chat_utils._parse_chat_message_content_parts._nemo_speechlm_exact_training_prompt


def test_install_prompt_contract_falls_back_to_vllm_entrypoints_chat_utils(monkeypatch):
    chat_utils, _ = _fake_vllm_chat_utils()
    monkeypatch.setitem(sys.modules, "vllm.renderers", None)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", SimpleNamespace(chat_utils=chat_utils))

    install_prompt_contract()

    assert chat_utils._parse_chat_message_content_parts._nemo_speechlm_exact_training_prompt


def test_install_prompt_contract_patches_vllm_0_33_chat_message(monkeypatch):
    """vLLM 0.33 made ``vllm.renderers.chat_utils`` a package with a public parser in ``chat_message``."""
    chat_message, calls = _fake_vllm_chat_utils("parse_chat_message_content_parts")
    # The 0.33 package re-exports the public parser but no longer has the private 0.32 name.
    package = SimpleNamespace(chat_message=chat_message, parse_chat_message_content_parts=object())
    monkeypatch.setitem(sys.modules, "vllm.renderers.chat_utils", package)
    monkeypatch.setitem(sys.modules, "vllm.renderers", SimpleNamespace(chat_utils=package))

    install_prompt_contract()

    parser = chat_message.parse_chat_message_content_parts
    assert parser._nemo_speechlm_exact_training_prompt
    assert not hasattr(package, "_parse_chat_message_content_parts")
    result = parser(
        "user",
        [_AUDIO, {"type": "text", "text": "Transcribe"}],
        _tracker("nemotron3_5_transcribe_mtp"),
        wrap_dicts=False,
        interleave_strings=False,
    )
    assert result == [{"role": "user", "content": "Transcribe <|audio|>"}]
    assert calls[0]["registered_audio_parts"] == [_AUDIO]


def test_install_prompt_contract_is_idempotent_on_vllm_0_33(monkeypatch):
    chat_message, _ = _fake_vllm_chat_utils("parse_chat_message_content_parts")
    monkeypatch.setitem(sys.modules, "vllm.renderers.chat_utils", SimpleNamespace(chat_message=chat_message))

    install_prompt_contract()
    patched = chat_message.parse_chat_message_content_parts
    install_prompt_contract()

    assert chat_message.parse_chat_message_content_parts is patched


def test_adapter_names_the_missing_public_parser():
    with pytest.raises(TypeError, match="parse_chat_message_content_parts is missing"):
        _patch_speechlm_training_prompt_contract(SimpleNamespace(), "parse_chat_message_content_parts")


class TestMarlinGatelessMoePadding:
    """Padding of vLLM's MoE w13 weights handles gate-less (1 shard) and gated (2 shards) experts."""

    INT_WNA16 = "vllm.model_executor.layers.fused_moe.oracle.int_wna16"

    def test_gateless_weight_pads_single_shard_columns(self):
        x = torch.arange(1, 1 + 2 * 3 * 928, dtype=torch.float32).reshape(2, 3, 928)

        padded = _pad_w13_shard_cols(x, 928, 960)

        assert padded.shape == (2, 3, 960)
        assert torch.equal(padded[..., :928], x)
        assert torch.count_nonzero(padded[..., 928:]) == 0

    def test_nemotron_geometry_from_the_load_failure_is_accepted(self):
        # shape '[128, 672, 2, 928]' is invalid for input of size 79822848 was the original error.
        x = torch.empty(128, 672, 928, dtype=torch.int32)

        assert _pad_w13_shard_cols(x, 928, 960).shape == (128, 672, 960)

    def test_gated_weight_pads_each_shard_and_keeps_order(self):
        unit = 4
        gate = torch.full((2, 3, unit), 1.0)
        up = torch.full((2, 3, unit), 2.0)

        padded = _pad_w13_shard_cols(torch.cat([gate, up], dim=-1), unit, 8)

        assert padded.shape == (2, 3, 16)
        assert torch.equal(padded[..., :4], gate)
        assert torch.equal(padded[..., 8:12], up)
        assert torch.count_nonzero(padded[..., 4:8]) == 0
        assert torch.count_nonzero(padded[..., 12:16]) == 0

    def test_equal_sizes_return_the_tensor_untouched(self):
        x = torch.zeros(2, 3, 7)
        assert _pad_w13_shard_cols(x, 7, 7) is x
        b = torch.zeros(2, 7)
        assert _pad_w13_bias(b, 7, 7) is b

    @pytest.mark.parametrize("columns", [3, 5, 12])
    def test_weight_with_neither_one_nor_two_shards_is_rejected(self, columns):
        with pytest.raises(ValueError, match="neither one nor two shards"):
            _pad_w13_shard_cols(torch.zeros(1, 1, columns), 4, 8)

    def test_bias_pads_one_and_two_shards(self):
        one = torch.ones(2, 4)
        padded = _pad_w13_bias(one, 4, 8)
        assert padded.shape == (2, 8) and torch.count_nonzero(padded[:, 4:]) == 0

        two = torch.cat([torch.full((2, 4), 1.0), torch.full((2, 4), 2.0)], dim=-1)
        padded = _pad_w13_bias(two, 4, 8)
        assert padded.shape == (2, 16)
        assert torch.equal(padded[:, :4], two[:, :4])
        assert torch.equal(padded[:, 8:12], two[:, 4:])
        assert torch.count_nonzero(padded[:, 4:8]) == 0 and torch.count_nonzero(padded[:, 12:]) == 0

    def test_install_replaces_both_helpers(self, monkeypatch):
        fake = SimpleNamespace(_pad_w13_shard_cols=lambda *_: None, _pad_w13_bias=lambda *_: None)
        monkeypatch.setitem(sys.modules, self.INT_WNA16, fake)

        install_marlin_gateless_moe_padding()

        assert fake._pad_w13_shard_cols is _pad_w13_shard_cols
        assert fake._pad_w13_bias is _pad_w13_bias

    def test_install_leaves_a_vllm_without_these_helpers_alone(self, monkeypatch):
        fake = SimpleNamespace()
        monkeypatch.setitem(sys.modules, self.INT_WNA16, fake)

        install_marlin_gateless_moe_padding()

        assert not hasattr(fake, "_pad_w13_shard_cols")

    def test_install_survives_a_vllm_without_the_module(self, monkeypatch):
        monkeypatch.setitem(sys.modules, self.INT_WNA16, None)

        install_marlin_gateless_moe_padding()

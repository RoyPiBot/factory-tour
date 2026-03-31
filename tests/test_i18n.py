"""
test_i18n.py - 測試多語言模組
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from i18n import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    get_prompt,
    SUPERVISOR_PROMPTS,
    TOUR_GUIDE_PROMPTS,
    SAFETY_EXPERT_PROMPTS,
    TECH_EXPERT_PROMPTS,
    QA_AGENT_PROMPTS,
    UI_STRINGS,
    LANGUAGE_NAMES,
)


class TestI18nBasics:
    def test_supported_languages(self):
        assert "zh-TW" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES
        assert "ja" in SUPPORTED_LANGUAGES

    def test_default_language(self):
        assert DEFAULT_LANGUAGE == "zh-TW"

    def test_language_names(self):
        assert LANGUAGE_NAMES["zh-TW"] == "繁體中文"
        assert LANGUAGE_NAMES["en"] == "English"
        assert LANGUAGE_NAMES["ja"] == "日本語"


class TestGetPrompt:
    def test_get_existing_language(self):
        prompt = get_prompt(SUPERVISOR_PROMPTS, "en")
        assert "supervisor" in prompt.lower() or "factory" in prompt.lower()

    def test_fallback_to_default(self):
        prompt = get_prompt(SUPERVISOR_PROMPTS, "ko")  # 不支援的語言
        assert prompt == SUPERVISOR_PROMPTS["zh-TW"]


class TestAllPromptsHaveAllLanguages:
    """確保每組 prompt 都有三種語言"""

    def _check_all_langs(self, prompt_dict, name):
        for lang in SUPPORTED_LANGUAGES:
            assert lang in prompt_dict, f"{name} 缺少 {lang}"
            assert len(prompt_dict[lang]) > 0, f"{name}[{lang}] 為空"

    def test_supervisor_prompts(self):
        self._check_all_langs(SUPERVISOR_PROMPTS, "SUPERVISOR_PROMPTS")

    def test_tour_guide_prompts(self):
        self._check_all_langs(TOUR_GUIDE_PROMPTS, "TOUR_GUIDE_PROMPTS")

    def test_safety_expert_prompts(self):
        self._check_all_langs(SAFETY_EXPERT_PROMPTS, "SAFETY_EXPERT_PROMPTS")

    def test_tech_expert_prompts(self):
        self._check_all_langs(TECH_EXPERT_PROMPTS, "TECH_EXPERT_PROMPTS")

    def test_qa_agent_prompts(self):
        self._check_all_langs(QA_AGENT_PROMPTS, "QA_AGENT_PROMPTS")


class TestUIStrings:
    """UI 翻譯字串測試"""

    def test_all_languages_have_ui_strings(self):
        for lang in SUPPORTED_LANGUAGES:
            assert lang in UI_STRINGS, f"UI_STRINGS 缺少 {lang}"

    def test_all_ui_keys_consistent(self):
        """所有語言應有相同的 key"""
        zh_keys = set(UI_STRINGS["zh-TW"].keys())
        for lang in ["en", "ja"]:
            lang_keys = set(UI_STRINGS[lang].keys())
            missing = zh_keys - lang_keys
            assert not missing, f"{lang} 缺少 key: {missing}"

    def test_ui_strings_not_empty(self):
        for lang in SUPPORTED_LANGUAGES:
            for key, value in UI_STRINGS[lang].items():
                assert len(value) > 0, f"UI_STRINGS[{lang}][{key}] 為空"

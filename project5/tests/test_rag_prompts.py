from __future__ import annotations

from conftest import load_src_module


def test_prompt_templates_include_required_placeholders() -> None:
    module = load_src_module("rag_prompts.py", "rag_prompts_mod")

    text_template = module.rag_text_qa_template_str()
    refine_template = module.rag_refine_template_str()

    assert "{context_str}" in text_template
    assert "{query_str}" in text_template
    assert "{existing_answer}" in refine_template
    assert "{context_msg}" in refine_template
    assert "{query_str}" in refine_template
    assert "ESG report QA analyst" in module.ESG_SYSTEM_PROMPT

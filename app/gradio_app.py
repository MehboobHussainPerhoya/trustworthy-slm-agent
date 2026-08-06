"""
app.py
Gradio interface for the Trustworthy SLM Agent — combines retrieval,
generation, and hallucination checking into a single chat-style demo.
Entry point for Hugging Face Spaces deployment (CPU tier by default).
"""

import gradio as gr
from agent import Agent

print("Initializing agent (this happens once, at startup)...")
agent = Agent()
print("Agent ready.")


def respond(question: str):
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    result = agent.ask(question)

    verdict = result["verdict"]
    if verdict == "supported":
        verdict_html = (
            '<div style="padding:10px;border-radius:8px;background:#e6f4ea;'
            'color:#1e7e34;font-weight:600;">✓ SUPPORTED — grounded in the retrieved sources</div>'
        )
    elif verdict == "partially_supported":
        verdict_html = (
            '<div style="padding:10px;border-radius:8px;background:#fff8e1;'
            'color:#8a6d00;font-weight:600;">⚠ PARTIALLY SUPPORTED — some claims not directly confirmed by sources</div>'
        )
    else:
        verdict_html = (
            '<div style="padding:10px;border-radius:8px;background:#fdecea;'
            'color:#a32020;font-weight:600;">✗ UNSUPPORTED — may contradict or go beyond the retrieved sources</div>'
        )

    sources_md = "\n\n".join(
        f"**[{s['section']}]** (relevance: {s['score']})\n\n{s['text']}"
        for s in result["sources"]
    )

    return result["answer"], verdict_html, sources_md


EXAMPLE_QUESTIONS = [
    "Why do language models hallucinate according to this paper?",
    "What is the singleton rate?",
    "Does RAG fully solve hallucination?",
    "What is the capital of France?",  # deliberately out-of-scope, demonstrates abstention
]

with gr.Blocks(title="Trustworthy SLM Agent") as demo:
    gr.Markdown(
        "# Trustworthy SLM Agent\n"
        "Ask questions about *\"Why Language Models Hallucinate\"* "
        "(Kalai, Nachum, Vempala, Zhang, 2025). Answers are grounded in "
        "retrieved excerpts from the paper and automatically checked for "
        "hallucination before being shown to you.\n\n"
        "*Note: this is a small, narrow-domain research demo — it will "
        "correctly decline or hedge on questions outside this paper's scope.*"
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="Your question", placeholder="Ask something about the paper...", scale=4
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_input)

    answer_output = gr.Textbox(label="Answer", lines=4)
    verdict_output = gr.HTML(label="Hallucination Check Verdict")
    sources_output = gr.Markdown(label="Retrieved Sources")

    submit_btn.click(
        fn=respond,
        inputs=question_input,
        outputs=[answer_output, verdict_output, sources_output],
    )
    question_input.submit(
        fn=respond,
        inputs=question_input,
        outputs=[answer_output, verdict_output, sources_output],
    )

    gr.Markdown(
        "---\n"
        "Built as a demonstration of responsible AI deployment: fine-tuned SLM "
        "+ retrieval grounding + automated hallucination detection + bias auditing. "
        "[View full model card and source](https://github.com/MehboobHussainPerhoya/trustworthy-slm-agent)"
    )

if __name__ == "__main__":
    demo.launch()
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

print("Loading model..")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    f"models/flan-t5-large-10ep/checkpoint-42000"
).to("cuda")


def get_responses(prompt, output_prefix=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    optional_kwargs = {}

    if output_prefix is not None:
        output_prefix = tokenizer.encode(
            "<pad> " + output_prefix, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")
        optional_kwargs["decoder_input_ids"] = output_prefix

    out = model.generate(
        inputs=input_ids,
        top_p=0.9,
        repetition_penalty=2.0,
        num_return_sequences=5,
        do_sample=True,
        max_length=200,
        **optional_kwargs,
    )

    return tokenizer.batch_decode(out, skip_special_tokens=True)


def clean_copy_pasted_history(history):
    # Remove the timestamps from the history

    return re.sub(r", \[.*\]", "", history)


def create_prompt(history):
    return f"""Respond to chat as Kyle Corbitt:

Recent messages:

{clean_copy_pasted_history(history)}

Kyle Corbitt:"""


@cache
def gradio_fn(history, current_prefix=None):
    prompt = create_prompt(history)
    responses = get_responses(prompt, output_prefix=current_prefix.strip())

    print(f"Finished for {current_prefix}")
    return "".join([f" - {response}\n" for response in responses])


example_history = """Alex Chung, [Jan 10, 2023 at 7:10:19 AM]:
Just a heads up, planning to do 1:1 around stand up time rather than in the AM

Kyle Corbitt, [Jan 10, 2023 at 10:03:39 AM]:
lmk when you want to get on

Alex Chung, [Jan 10, 2023 at 10:13:13 AM]:
Didn't we move stand up to 3pm pacific?

Kyle Corbitt, [Jan 10, 2023 at 10:13:18 AM]:
ah yeah

I thought "around standup time" meant around normal standup time 10am

but got it now

Alex Chung, [Jan 10, 2023 at 10:13:37 AM]:
Sorry!!!!"""

print("Starting gradio..")

with gr.Blocks() as demo:
    gr.Markdown(
        """
## Chat History Completion

Use a T5 model trained on Kyle's Telegram chat history to form a response.
"""
    )

    with gr.Column():
        history = gr.components.Textbox(
            lines=10, label="Chat History", value=example_history
        )
        prefix = gr.components.Textbox(
            lines=1,
            label="Current Prefix",
            placeholder="Start typing here and we'll try to complete it...",
        )

    with gr.Column():
        outputs = gr.components.Textbox(label="Completions")

    # demo.load(gradio_fn, [history, prefix], outputs, every=0.1)
    prefix.change(gradio_fn, [history, prefix], outputs, every=0.1)
    # history.change(gradio_fn, [history, prefix], outputs)
    # prefix.change(gradio_fn, [history, prefix], outputs, every=1)

demo.queue().launch(share=True, server_port=7860)

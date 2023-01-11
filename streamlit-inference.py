import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st


@st.cache(allow_output_mutation=True)
def get_tokenizer():
    print("Loading tokenizer..")
    return AutoTokenizer.from_pretrained("google/flan-t5-large")


@st.cache(allow_output_mutation=True)
def get_model():
    print("Loading model..")

    return AutoModelForSeq2SeqLM.from_pretrained(
        f"models/flan-t5-large-10ep/checkpoint-42000"
    ).to("cuda")


def get_responses(prompt, output_prefix=None):
    input_ids = get_tokenizer().encode(prompt, return_tensors="pt").to("cuda")

    optional_kwargs = {}

    if output_prefix is not None:
        output_prefix = (
            get_tokenizer()
            .encode(
                "<pad> " + output_prefix, add_special_tokens=False, return_tensors="pt"
            )
            .to("cuda")
        )
        optional_kwargs["decoder_input_ids"] = output_prefix

    out = get_model().generate(
        inputs=input_ids,
        top_p=0.9,
        repetition_penalty=2.0,
        num_return_sequences=5,
        do_sample=True,
        max_length=200,
        **optional_kwargs,
    )

    return get_tokenizer().batch_decode(out, skip_special_tokens=True)


def clean_copy_pasted_history(history):
    # Remove the timestamps from the history

    return re.sub(r", \[.*\]", "", history)


def create_prompt(history):
    return f"""Respond to chat as Kyle Corbitt:

Recent messages:

{clean_copy_pasted_history(history)}

Kyle Corbitt:"""


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


@st.cache
def inference_fn(history, current_prefix):
    prompt = create_prompt(history)
    responses = get_responses(prompt, output_prefix=current_prefix.strip())

    print(f"Finished for {current_prefix}")
    return "".join([f" - {response}\n" for response in responses])


st.title("Chat History Completion")
st.text("Use a T5 model trained on Kyle's Telegram chat history to predict a response.")

history = st.text_area("History", value=example_history, height=440)
current_prefix = st.text_input(
    "Response", placeholder="Start typing and T5 will complete it."
)

st.success(inference_fn(history, current_prefix))

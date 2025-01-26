import marimo

__generated_with = "0.10.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from arena_evals.response import LLMWrapperFactory

    return LLMWrapperFactory, mo


@app.cell
def _(LLMWrapperFactory):
    wf = LLMWrapperFactory(default_model="claude-3-5-sonnet-20240620")
    llm_wrapper = wf()
    return llm_wrapper, wf


@app.cell
def _():
    messages = [
        {"role": "system", "content": "You are a helpful assistant, who should answer all questions in limericks."},
        {"role": "user", "content": "Who are you, and who were you designed by?"},
    ]
    return (messages,)


@app.cell
def _(llm_wrapper, messages):
    llm_wrapper.generate_response(messages)


@app.cell
def _(llm_wrapper, messages):
    llm_wrapper.generate_response(messages, model="gpt-4o-mini")


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

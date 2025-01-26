import marimo

__generated_with = "0.10.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from arena_evals.response import OpenAIWrapperFactory

    return OpenAIWrapperFactory, mo


@app.cell
def _(OpenAIWrapperFactory):
    oaiwf = OpenAIWrapperFactory()
    oai = oaiwf()
    return oai, oaiwf


@app.cell
def _():
    messages = [
        {"role": "system", "content": "You are a helpful assistant, who should answer all questions in limericks."},
        {"role": "user", "content": "Who are you, and who were you designed by?"},
    ]
    return (messages,)


@app.cell
def _(messages, oai):
    oai.generate_response(messages)


if __name__ == "__main__":
    app.run()

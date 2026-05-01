# Malaya Tagging Pipeline

This is a Malay pipeline that allows you to tokenizer, lemmatize, and Part Of Speech (POS) tag Malay text. It is a light weight wrapper around [malaysia-ai/malaya](https://github.com/malaysia-ai/malaya) whereby this wrapper provides:
* Support for Python>3.10
* Lemmatize and POS tags tokens rather than strings of text, this is useful when you want token level lemmas and POS tags that align.
* POS tags all punctuation tokens as defined by `malaya.text.function.PUNCTUATION` (!"#$%&\'()*+,./:;<=>?@[\\]^_\`{|}~) as `PUNCT` POS tags, this is a rule based decision that is hard coded in this codebase.
* The lemmatizer and POS tagger models, which are PyTorch models, are set to evaluation mode.
* The lemmatizer and POS tagger models are ran within a PyTorch inference context manager which speeds up and reduces the memory usage of the model.

**NOTE** the POS tagger outputs tags from the [Universal POS tagset](https://universaldependencies.org/u/pos/) as well as a `PUNCT` tag representing punctuation.

## Install

Using Pip (installs the tag `v0.1.0`):
``` bash
pip install "malaya-tagging-pipeline @ git+https://github.com/UCREL/malaya-tagging-pipeline.git@e0cdbc9158e7d14549ed44c5d83a933a2b3976a9"
```

Using UV (installs the tag `v0.1.0`):
``` bash
uv add --optional malay "malaya-tagging-pipeline==0.1.0" git+https://github.com/UCREL/malaya-tagging-pipeline@e0cdbc9158e7d14549ed44c5d83a933a2b3976a9
```

## Example

The example below shows how to sentence split, tokenize, lemmatize, and POS tag Malay data (**NOTE**: this will download 2 HuggingFace models for lemmatization and POS tagging, both of these models are less than 200MB):

``` python
import malaya

from malaya_tagging_pipeline import stem_tokens, tag_tokens, word_tokenize

# This text has been taken from the abstract of the Prostate Cancer Wikipedia page
# https://ms.wikipedia.org/wiki/Barah_prostat
test_text = """
Barah prostat atau kanser prostat ialah satu bentuk kanser yang berkembang di dalam prostat, satu kelenjar dalam sistem pembiakan jantan atau lelaki. Ia merupakan salah satu kanser yang paling biasa di kalangan lelaki, terutamanya yang berumur lebih 50 tahun.
"""

sentence_splitter = malaya.tokenizer.SentenceTokenizer()
word_tokenizer = malaya.tokenizer.Tokenizer()
lemmatizer = malaya.stem.huggingface('mesolitica/stem-lstm-512', force_check=True)
pos_tagger = malaya.pos.huggingface("mesolitica/pos-t5-small-standard-bahasa-cased", force_check=True)
sentence_splits = sentence_splitter.tokenize(test_text)

for sentence_index, sentence in enumerate(sentence_splits):
    print(f"Sentence: {sentence_index + 1}")
    tokens = word_tokenize(word_tokenizer, sentence, lowercase=False)
    lemmas  = stem_tokens(lemmatizer, tokens)
    pos_tags = tag_tokens(pos_tagger, tokens)

    for token, lemma, pos_tag in zip(tokens, lemmas, pos_tags):
        print(f"{token}\t{lemma}\t{pos_tag}")
    print("------"*5)
    print()
```

Output:

``` python
Sentence: 1
Barah   Barah   PROPN
prostat ostat   NOUN
atau    atau    CCONJ
kanser  kanser  NOUN
prostat ostat   NOUN
ialah   ialah   AUX
satu    satu    DET
bentuk  bentuk  NOUN
kanser  kanser  NOUN
yang    yang    PRON
berkembang      kembang VERB
di      k       ADP
dalam   dalam   ADP
prostat ostat   NOUN
,       ,       PUNCT
satu    satu    DET
kelenjar        kelenjar        NOUN
dalam   dalam   ADP
sistem  sistem  NOUN
pembiakan       biak    NOUN
jantan  jantan  NOUN
atau    atau    CCONJ
lelaki  lelaki  NOUN
.       .       PUNCT
------------------------------

Sentence: 2
Ia      Ia      PRON
merupakan       rupa    VERB
salah   salah   DET
satu    satu    DET
kanser  kanser  NOUN
yang    yang    PRON
paling  paling  ADV
biasa   biasa   ADJ
di      di      ADP
kalangan        kalang  NOUN
lelaki  lelaki  NOUN
,       ,       PUNCT
terutamanya     utama   ADV
yang    yang    PRON
berumur umur    VERB
lebih   lebih   ADV
50      as      NUM
tahun   tahun   NOUN
.       .       PUNCT
------------------------------
```


## Setup

You can either use the dev container with your favourite editor, e.g. VSCode. Or you can create your setup locally below we demonstrate both.

In both cases they share the same tools, of which these tools are:
* [uv](https://docs.astral.sh/uv/) for Python packaging and development
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.

### Dev Container

A [dev container](https://containers.dev/) uses a docker container to create the required development environment, the Dockerfile we use for this dev container can be found at [./.devcontainer/Dockerfile](./.devcontainer/Dockerfile). To run it locally it requires docker to be installed, you can also run it in a cloud based code editor, for a list of supported editors/cloud editors see [the following webpage.](https://containers.dev/supporting)

To run for the first time on a local VSCode editor (a slightly more detailed and better guide on the [VSCode website](https://code.visualstudio.com/docs/devcontainers/tutorial)):
1. Ensure docker is running.
2. Ensure the VSCode [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension is installed in your VSCode editor.
3. Open the command pallete `CMD + SHIFT + P` and then select `Dev Containers: Rebuild and Reopen in Container`

You should now have everything you need to develop, `uv`, `make`, for VSCode various extensions like `Pylance`, etc.

If you have any trouble see the [VSCode website.](https://code.visualstudio.com/docs/devcontainers/tutorial).

### Local

To run locally first ensure you have the following tools installted locally:
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python packaging and development. (version `0.9.6`)
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.
  * Ubuntu: `apt-get install make`
  * Mac: [Xcode command line tools](https://mac.install.guide/commandlinetools/4) includes `make` else you can use [brew.](https://formulae.brew.sh/formula/make)
  * Windows: Various solutions proposed in this [blog post](https://earthly.dev/blog/makefiles-on-windows/) on how to install on Windows, inclduing `Cygwin`, and `Windows Subsystem for Linux`.

When developing on the project you will want to install the Python package locally in editable format with all the extra requirements, this can be done like so:

```bash
uv sync
```

### Linting

Linting and formatting with [ruff](https://docs.astral.sh/ruff/) it is a replacement for tools like Flake8, isort, Black etc, and we us [ty](https://github.com/astral-sh/ty) for type checking.

To run the linting:

``` bash
make lint
```

### Tests

To run the tests (uses pytest and coverage) and generate a coverage report:

``` bash
make test
```
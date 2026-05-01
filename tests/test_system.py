from typing import cast

import malaya
from transformers import PreTrainedTokenizerFast

from malaya_tagging_pipeline import stem_tokens, tag_tokens, word_tokenize

# This text has been taken from the abstract of the Prostate Cancer Wikipedia page
# https://ms.wikipedia.org/wiki/Barah_prostat
test_text = """
Barah prostat atau kanser prostat ialah satu bentuk kanser yang berkembang di dalam prostat, satu kelenjar dalam sistem pembiakan jantan atau lelaki. Ia merupakan salah satu kanser yang paling biasa di kalangan lelaki, terutamanya yang berumur lebih 50 tahun.
"""

expected_output = [
"""
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
.       .       PUNCT""",
"""
Ia      Ia      PRON
merupakan       rupa    VERB
salah   salah   DET
satu    satu    DET
kanser  kanser  NOUN
yang    yang    PRON
paling  paling  ADV
biasa   biasa   ADJ
di      k      ADP
kalangan        kalang  NOUN
lelaki  lelaki  NOUN
,       ,       PUNCT
terutamanya     utama   ADV
yang    yang    PRON
berumur umur    VERB
lebih   lebih   ADV
50      af      NUM
tahun   tahun   NOUN
.       .       PUNCT
"""
]

long_test_text = """
Radioterapi atau terapi sinaran ialah satu kaedah perubatan untuk merawat penyakit Barah. Ia menggunakan sinaran pengion seperti sinar-X, sinar gama atau elektron, untuk mengubati atau mengawal sel-sel barah. Perkataan tersebut terdiri daripada gabungan “radio” bermaksud sinaran dan “terapi” bermaksud rawatan.
Radioterapi boleh digunakan kerana sinaran atau radiasi menghentikan barah daripada membesar dan sel barah daripada membahagi. Ia diarahkan kepada sel barah dengan dua cara- radiasi dalaman dan radiasi luaran.
Radiasi luaran bermakna radioterapi diberi secara luar badan daripada sebuah mesin yang mengarahkan sinaran kepada ketumbuhan barah. Ia adalah seperti mengambil sinar-X tetapi dosnya jauh lebih tinggi daripada sinar-x untuk diagnosis. Terdapat beberapa jenis mesin radioterapi yang berbeza. Ini termasuk pemecut linear dan mesin kobalt.
Radiasi dalaman bermakna radioterapi diberi dengan meletak sebuah bekas radioaktif, dawai radioaktif yang halus atau jarum yang diletak dalam badan dekat dengan atau di tempat barah wujud. Ini dipanggil sebagai implan radioaktif. Implan tersebut mengandungi bahan radioaktif seperti yttrium, iridium atau caesium. Implan tersebut mengarahkan sinaran radiasi kepada barah tersebut.
Radiasi luaran daripada mesin biasa digunakan untuk rawatan radioterapi. Pakar terapi radiasi ataupun pakar onkologi akan menggunakan radiasi dalaman jika sel barah berada dekat dengan permukaan badan atau tempat yang senang untuk dimasuki. Mereka boleh menggunakan implan untuk barah pay udara, barah serviks, dan barah lidah. Jika radiasi dalaman digunakan, biasanya radiasi luaran pun akan digunakan juga. Walau bagaimanapun, pada hari ini, semakin banyak rawatan radiasi yang bersasaran diberi dengan wujudnya radiologi intervensi yang telah menjadi satu sumbangan besar kepada rawatan radiasi.
Jenis rawatan yang digunakan bergantung kepada jenis barah yang dihidapi, tempat barah wujud dan saiz tumbuhan tersebut. Rawatan perlu dikhaskan mengikut setiap individu. Pakar Radioterapi/Pakar Onkologi hanya akan menggunakan rawatan radioterapi jika rawatan tersebut membawa lebih kebaikan daripada keburukan.
Biasanya radioterapi digunakan bagi merawat tumor kanser kerana keupayaannya untuk mengawal pertumbuhan sel. Radiasi pengionan bertindak dengan merosakkan DNA tisu barah yang membawa kepada kematian selular. Untuk menyelamatkan tisu normal (seperti kulit atau organ Yang mesti dilalui oleh radiasi bagi merawat tumor), bentuk pancaran radiasi disasar dari beberapa sudut dedahan agar bersilang pada tumor, memberikan dos yang lebih besar terserap di situ berbanding pada tisu yang sihat di kawasan sekitar.Selain tumur itu sendiri, medan sinaran mungkin membabitkan penyaliran nod limfa jika ia secara klinikal atau radiologinya terbabit dengan tumor, atau jika ada dianggap sebagai berisiko bagi penyebaran malignan subklinikal. Ia adalah perlu untuk membenarkan ketidak pastian kedudukan harian dan pergerakan tumor dalaman. Ketidaktentuan ini boleh disebabkan oleh pergerakan dalaman (contohnya, pernafasan dan pengisian pundi kencing) dan pergerakan kulit luar berbanding pada kedudukan tumor.
Malangnya, sinaran radiasi tidak boleh membezakan sel barah daripada sel manusia yang biasa. Walau bagaimanapun, sel barah tidak boleh menahan kesan radiasi dan akan mati. Sel manusia yang biasa akan pulih semula dan kesan tetap tidak akan dialami. Di sebab sel barah membesar lebih cepat berbanding dengan sel biasa, ia akan lebih terjejas dengan radiasi. Sel biasa juga akan terjejas dan mungkin menyebabkan kesan sampingan pada badan manusia.
"""

def expected_sentence_to_token_data(sentence: str) -> list[tuple[str, str, str]]:
    sentence_token_data = sentence.split("\n")
    expected_token_data: list[tuple[str, str, str]] = []
    for token_data in sentence_token_data:
        token_data = token_data.strip()
        if not token_data:
            continue
        token, lemma, pos_tag = token_data.split()
        expected_token_data.append((token, lemma, pos_tag))
    return expected_token_data


def test_pos_tagger_tokenizer() -> None:
    """
    Testing as with version < 5 of transformers the tokenizer is not created correctly,
    it used the Fast version I believe of the T5 Tokenizer which is not compitable with this model and in 
    doing so created many sub word tokens per token, thus here we check that the number of sub word tokens 
    created is correct.
    """
    pos_tagger = malaya.pos.huggingface("mesolitica/pos-t5-small-standard-bahasa-cased", force_check=True)
    pos_tagger_tokenizer = pos_tagger.tokenizer
    pos_tagger_tokenizer = cast(PreTrainedTokenizerFast, pos_tagger_tokenizer)
    expected_number_sub_word_tokens = [30, 22]
    for sentence_index, sentence in enumerate(expected_output):
        sentence_tokens = [token for (token, _, _) in expected_sentence_to_token_data(sentence)]
        pos_tagger_sub_word_tokens = pos_tagger_tokenizer(sentence_tokens, truncation=False, padding=False, is_split_into_words=True, return_tensors='pt')
        assert pos_tagger_sub_word_tokens.input_ids.shape == (1, expected_number_sub_word_tokens[sentence_index])

def test_system() -> None:
    sentence_splitter = malaya.tokenizer.SentenceTokenizer()
    word_tokenizer = malaya.tokenizer.Tokenizer()
    lemmatizer = malaya.stem.huggingface('mesolitica/stem-lstm-512', force_check=True)
    pos_tagger = malaya.pos.huggingface("mesolitica/pos-t5-small-standard-bahasa-cased", force_check=True)
    sentence_splits = sentence_splitter.tokenize(test_text)
    assert len(sentence_splits) == len(expected_output)
    for sentence_index, sentence in enumerate(sentence_splits):
        expected_sentence = expected_output[sentence_index]
        expected_token_data = expected_sentence_to_token_data(expected_sentence)

        tokens = word_tokenize(word_tokenizer, sentence, lowercase=False)
        lemmas  = stem_tokens(lemmatizer, tokens)
        pos_tags = tag_tokens(pos_tagger, tokens)

        assert len(tokens) == len(expected_token_data)
        assert len(lemmas) == len(expected_token_data)
        assert len(pos_tags) == len(expected_token_data)

        for token_index, (token, lemma, pos_tag) in enumerate(zip(tokens, lemmas, pos_tags)):
            assert token == expected_token_data[token_index][0]
            assert lemma == expected_token_data[token_index][1]
            assert pos_tag == expected_token_data[token_index][2]
            

def test_long_text() -> None:
    """
    Testing as their has been issues with long texts failing with the POS Tagger.
    """
    sentence_splitter = malaya.tokenizer.SentenceTokenizer()
    word_tokenizer = malaya.tokenizer.Tokenizer()
    lemmatizer = malaya.stem.huggingface('mesolitica/stem-lstm-512', force_check=True)
    pos_tagger = malaya.pos.huggingface("mesolitica/pos-t5-small-standard-bahasa-cased", force_check=True)
    sentence_splits = sentence_splitter.tokenize(test_text)
    
    for sentence in sentence_splits:
        tokens = word_tokenize(word_tokenizer, sentence, lowercase=False)
        _  = stem_tokens(lemmatizer, tokens)
        _ = tag_tokens(pos_tagger, tokens)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from langdetect import detect
from memory.log_util import log_exception
import nltk

nltk.download('punkt')

lang_map = {
    'ar': 'arabic',
    'cs': 'czech',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'he': 'hebrew',
    'it': 'italian',
    'ja': 'japanese',
    'pt': 'portuguese',
    'sk': 'slovak',
    'uk': 'ukrainian'
}

def get_sumy_language(text):
    detected_language = detect(text)
    if detected_language in lang_map:
        return lang_map[detected_language]
    return 'english'

def summarize_text_basic(text, sentence_count = 3):
    language = get_sumy_language(text)

    parser = PlaintextParser.from_string(text, Tokenizer(language))

    stemmer = Stemmer(language)
    summarizer = EdmundsonSummarizer(stemmer)
    stop_words = get_stop_words(language)

    summarizer.null_words = stop_words
    summarizer.bonus_words = parser.significant_words
    summarizer.stigma_words = parser.stigma_words

    summary_sentences = summarizer(parser.document, sentence_count)
    summary = ' '.join([str(sentence) for sentence in summary_sentences])

    return summary

def summarize_text_with_gpt(text, openai_client, model_name):
    try:
        if len(text.split()) > 1024:
            # Crop the text to 1024 tokens
            text = ' '.join(text.split()[:1024])

        prompt = "Write a very short summary in the same language as the text:\n\n" + text

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=256
        )

        return response.choices[0].message.content
    except Exception:
        log_exception()
    
    return summarize_text_basic(text)
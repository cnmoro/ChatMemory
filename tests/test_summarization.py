
from memory.summarization import get_sumy_language, \
    summarize_text_basic, summarize_text_with_gpt
from openai import OpenAI
import os

OPENAI_KEY = os.environ.get('OPENAI_KEY')
openai_client = OpenAI(api_key=OPENAI_KEY)

sample_news_text = """
Luton led at half-time through Chiedozie Ogbene's header as Liverpool, and Luis Diaz in particular, spurned chance after chance.
The Reds were rampant after the break and their pressure paid off when captain Virgil van Dijk showed desire and determination to get his side on level terms by powering home a header from a 56th-minute corner.
Just 125 seconds later Liverpool led as Cody Gakpo nodded in from close range as Alexis Mac Allister fired in a cross from the byeline.
Diaz, with his ninth shot of the game, finally got his reward for relentless work with the third goal in the 71st minute.
The Colombian forward danced his way into the box and coolly side-footed beyond Thomas Kaminski at the near post to give the hosts some breathing space on what had initially looked like a night when they could slip up in the title race.
""".strip()

sample_news_text_small = """
Luton led at half-time through Chiedozie Ogbene's header as Liverpool, and Luis Diaz in particular, spurned chance after chance.
""".strip()

lorem_ipsum = """
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium,
totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae
dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit,
sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.
Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit,
sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.
""".strip()

def test_sumy_language_detection():
    assert get_sumy_language("This is a test") == 'english'

def test_failed_language_detection_english_fallback():
    assert get_sumy_language(lorem_ipsum) == 'english'

def test_sumy_summarization():
    assert len(summarize_text_basic(sample_news_text)) > 0

def test_summarizer_with_gpt_sumy_fallback():
    assert len(summarize_text_with_gpt(sample_news_text, None, 'undefined')) > 0

def test_summarizer_with_gpt():
        assert len(summarize_text_with_gpt(sample_news_text_small, openai_client, 'gpt-3.5-turbo')) > 0

def test_summarization_large_text_crop():
    big_text = " ".join([sample_news_text] * 100)
    assert len(summarize_text_with_gpt(big_text, openai_client, 'gpt-3.5-turbo')) > 0

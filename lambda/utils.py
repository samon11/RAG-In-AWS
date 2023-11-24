import os
import nltk
from nltk.tokenize import sent_tokenize
import fitz

nltk.data.path.append(os.getenv('LAMBDA_TASK_ROOT', './'))

def split_sents(text: str, group_size=3, prefix=''):
    sentences = sent_tokenize(text.replace('\n', ' ').replace('\r', ''))
    return [prefix + ' '.join(sentences[i:i+group_size]) for i in range(0, len(sentences), group_size)]

def parse_pdf(stream, max_len=3, split=True):
    pdf_document = fitz.open(stream=stream, filetype='pdf')
    total_pages = pdf_document.page_count
    text = ''
    for page_number in range(total_pages):
        page = pdf_document.load_page(page_number)
        page_text = page.get_text()
        text += page_text

    pdf_document.close()

    texts = split_sents(text, max_len)

    return texts if split else text

def parse_txt(text, max_len=4):
    return split_sents(text, max_len)

if __name__ == '__main__':
    print('Setting up NLTK...')
    nltk.download('punkt', download_dir=os.getenv('LAMBDA_TASK_ROOT', './'))

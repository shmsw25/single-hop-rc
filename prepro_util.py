import random
import bisect
import re
import spacy
import json
import tokenization

import numpy as np
from tqdm import tqdm
from gensim import corpora, models, similarities

from IPython import embed

nlp = spacy.blank("en")

def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return a[idx-1], d2
        else:
            return a[idx], d1

def fix_span(para, offsets, span):
    span = span.strip()
    parastr = "".join(para)
    assert span in parastr, '{}\t{}'.format(span, parastr)

    begins, ends = map(list, zip(*[y for x in offsets for y in x]))

    best_dist = 1e200
    best_indices = None

    if span == parastr:
        return parastr, (0, len(parastr)), 0

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()

        fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < end_offset)
        fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > begin_offset)

        if d1 + d2 < best_dist:
            best_dist = d1 + d2
            best_indices = (fixed_begin, fixed_end)
            if best_dist == 0:
                break

    assert best_indices is not None
    return parastr[best_indices[0]:best_indices[1]], best_indices, best_dist

def word_tokenize(sent, keep_capital=False):
    doc = nlp(sent)
    tokens = [token.text.strip() for token in doc]
    tokens = [token for token in tokens if len(token) > 0]
    if len(tokens) == 0:
        return None
    if tokens[-1].endswith('.') and tokens[-1] != '.':
        tokens[-1] = tokens[-1][:-1]
        tokens.append(".")
    return tokens

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        pre = current
        current = text.find(token, current)
        if current < 0:
            print (text, token, current)
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def find_span_from_text(context, tokens, answer):
    if answer.strip() in ['yes', 'no']:
        return [{'text': answer, 'answer_start': 0}]

    assert answer in context

    offset = 0
    spans = []
    scanning = None
    process = []

    for i, token in enumerate(tokens):
        while context[offset:offset+len(token)] != token:
            offset += 1
            if offset >= len(context):
                break
        if scanning is not None:
            end = offset + len(token)
            if answer.startswith(context[scanning[-1]:end]):
                if context[scanning[-1]:end] == answer:
                    spans.append(scanning[0])
                elif len(context[scanning[-1]:end]) >= len(answer):
                    scanning = None
            else:
                scanning = None
        if scanning is None and answer.startswith(token):
            if token == answer:
                spans.append(offset)
            if token != answer:
                scanning = [offset]
        offset += len(token)
        if offset >= len(context):
            break
        process.append((token, offset, scanning, spans))

    answers = []

    for span in spans:
        if context[span:span+len(answer)] != answer:
            print (context[span:span+len(answer)], answer)
            print (context)
            embed()
            assert False
        answers.append({'text': answer, 'answer_start': span})
    #if len(answers)==0:
    #    print ("*"*30)
    #    print (context, answer)
    return answers

def detect_span(_answers, context, doc_tokens, char_to_word_offset):
    orig_answer_texts = []
    start_positions = []
    end_positions = []
    switches = []

    if 'answer_start' not in _answers[0]:
        answers = []
        for answer in _answers:
            answers += find_span_from_text(context, doc_tokens, answer['text'])
    else:
        answers = _answers

    for answer in answers:
        orig_answer_text = answer["text"]
        answer_offset = answer["answer_start"]
        answer_length = len(orig_answer_text)

        if orig_answer_text in ["yes", "no"]:
            start_position, end_position = 0, 0
            switch = 1 if orig_answer_text == "yes" else 2
        else:
            switch = 0
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            #if actual_text.replace(' ', '').find(cleaned_answer_text.replace(' ', '')) == -1:
            #    print ("Could not find answer: '%s' vs. '%s'" % (actual_text, cleaned_answer_text))

        orig_answer_texts.append(orig_answer_text)
        start_positions.append(start_position)
        end_positions.append(end_position)
        switches.append(switch)

    return orig_answer_texts, switches, start_positions, end_positions

def retrieve_tfidf(doc_tokens, ques_token):
    # 2-d list of words
    dictionary = corpora.Dictionary(doc_tokens)
    corpus = [dictionary.doc2bow(doc_token) for doc_token in doc_tokens]
    tfidf = models.TfidfModel(corpus)
    index = similarities.MatrixSimilarity(tfidf[corpus])
    sims = index[tfidf[dictionary.doc2bow(ques_token)]]
    return [sim[0] for sim in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)]
    #return [sim[0] for sim in sorted(sims, key=lambda x: x[1], reverse=True)]


def prepro_tfidf(input_file, output_file):
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    with open(input_file, "r") as reader:
        input_data = json.load(reader)

    with open(output_file, "r") as reader:
        tfidf_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = {}
    tfidf_results = [[], [], [], [], [], [], [], []]

    for article in tqdm(input_data):
        paragraphs = article['context']
        sfs = [[_process_sent(t), s] for t, s in article['supporting_facts']]
        question = article['question']
        answer = article['answer'].strip()

        sentences, labels = [], []
        for para in paragraphs:
            title = para[0]
            content = para[1]
            for sent_idx, sent in enumerate(content):
                label = [_process_sent(title), sent_idx] in sfs
                '''
                paragraph_text = "{} {}".format(title.strip(), sent.strip())
                doc_tokens = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                sentences.append(doc_tokens)
                '''
                labels.append(int(label))
        idxs = tfidf_data[article['_id']]

        '''
        idxs = retrieve_tfidf(sentences, question.split(' '))
        examples[article['_id']] = idxs
        '''
        total = len([l for l in labels if l==1])
        for ki, k in enumerate([5, 10, 15, 20, 25, 30, 35, 40]):
            part = len([labels[i] for i in idxs[:k] if labels[i]==1])
            tfidf_results[ki].append(1.0*part/total)

    #with open(output_file, "w") as f:
    #    json.dump(examples, f)
    return tfidf_results

if __name__ == '__main__':
    data_dir = "/home/sewon/data/hotpotqa/"
    dev_results = prepro_tfidf(data_dir+"hotpot_dev_distractor_v1.json", \
                               data_dir+"dev_tfidf.json")
    train_results = prepro_tfidf(data_dir+"hotpot_train_v1.json", \
                                 data_dir+"train_tfidf.json")
    print ([np.mean(r) for r in dev_results])
    print ([np.mean(r) for r in train_results])


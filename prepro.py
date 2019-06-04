import json
import tokenization
import collections
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from prepro_util import detect_span, retrieve_tfidf
from hotpot_evaluate_v1 import f1_score as hotpot_f1_score
from Example import *

def get_dataloader(logger, args, input_file, is_training, \
                   batch_size, num_epochs, tokenizer):

    examples = read_squad_examples(
            input_file=input_file, is_training=is_training, debug=args.debug)

    num_train_steps = int(len(examples) / batch_size * num_epochs)

    train_features, n_answers_with_truncated_answers = convert_examples_to_features(
        logger=logger,
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        max_n_answers=args.max_n_answers if is_training else 1,
        is_training=is_training)

    logger.info("  Num orig examples = %d", len(examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", batch_size)
    if is_training:
        logger.info("  Num steps = %d", num_train_steps)
        logger.info("  %% of tuncated_answers = %.2f%%" % \
                    (100.0*n_answers_with_truncated_answers/len(train_features)))


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    if is_training:
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_switches = torch.tensor([f.switch for f in train_features], dtype=torch.long)
        all_answer_mask = torch.tensor([f.answer_mask for f in train_features], dtype=torch.long)
        assert all_start_positions.size() == all_end_positions.size() == \
            all_switches.size() == all_answer_mask.size()
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                    all_start_positions, all_end_positions, all_switches, all_answer_mask)
        sampler=RandomSampler(dataset)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                        all_example_index)
        sampler=SequentialSampler(dataset)


    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader, examples, train_features, num_train_steps


def read_squad_examples(input_file, is_training, debug):
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    input_data = []
    for _input_file in input_file.split(','):
        with open(_input_file, "r") as reader:
            this_data = json.load(reader)["data"]
            if debug:
                this_data = this_data[:20]
            input_data += this_data

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            context = paragraph['context']
            qas = paragraph['qas']
            if type(context)==str:
                context = [context]
                for qa in qas:
                    qa['answers'] = [qa['answers']]
            assert np.all([len(qa['answers'])==len(context) for qa in qas])
            doc_tokens_list, char_to_word_offset_list = [], []
            for paragraph_text in context:
                doc_tokens = []
                char_to_word_offset = []
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
                    char_to_word_offset.append(len(doc_tokens) - 1)
                doc_tokens_list.append(doc_tokens)
                char_to_word_offset_list.append(char_to_word_offset)

            for qa in qas:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                switch = 0

                assert len(qa['answers']) == len(context)
                if 'final_answers' in qa:
                    all_answers = qa['final_answers']
                else:
                    all_answers = []
                    for answers in qa['answers']:
                        all_answers += [a['text'] for a in answers]

                assert len(all_answers)>0

                original_answers_list, start_positions_list, end_positions_list, switches_list = [], [], [], []
                for (paragraph_text, doc_tokens, char_to_word_offset, answers) in zip( \
                        context, doc_tokens_list, char_to_word_offset_list, qa['answers']):

                    if is_training:
                        if len(answers)==0:
                            original_answers = [""]
                            start_positions, end_positions = [0], [0]
                            switches = [3]
                        else:
                            original_answers, switches, start_positions, end_positions = detect_span(\
                                answers, paragraph_text, doc_tokens, char_to_word_offset)
                    else:
                        original_answers, switches, start_positions, end_positions = [], [], [], []
                    original_answers_list.append(original_answers)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)
                    switches_list.append(switches)
                examples.append(SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens_list,
                        orig_answer_text=original_answers_list,
                        all_answers=all_answers,
                        start_position=start_positions_list,
                        end_position=end_positions_list,
                        switch=switches_list))

    return examples


def convert_examples_to_features(logger, examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, max_n_answers, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    truncated = []
    features = []
    features_with_truncated_answers = []

    for (example_index, example) in tqdm(enumerate(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        assert len(example.doc_tokens) == len(example.orig_answer_text) == \
            len(example.start_position) == len(example.end_position) == len(example.switch)

        for (doc_tokens, original_answer_text_list, start_position_list, end_position_list, switch_list) in \
                zip(example.doc_tokens, example.orig_answer_text, example.start_position, \
                    example.end_position, example.switch):

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_positions = []
            tok_end_positions = []

            if is_training:
                for (orig_answer_text, start_position, end_position) in zip( \
                            original_answer_text_list, start_position_list, end_position_list):
                    if orig_answer_text in ['yes', 'no']:
                        tok_start_positions.append(-1)
                        tok_end_positions.append(-1)
                        continue
                    tok_start_position = orig_to_tok_index[start_position]
                    if end_position < len(doc_tokens) - 1:
                        tok_end_position = orig_to_tok_index[end_position + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1
                    (tok_start_position, tok_end_position) = _improve_answer_span(
                        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                        orig_answer_text)
                    tok_start_positions.append(tok_start_position)
                    tok_end_positions.append(tok_end_position)
                to_be_same = [len(original_answer_text_list), \
                                    len(start_position_list), len(end_position_list),
                                    len(switch_list), \
                                    len(tok_start_positions), len(tok_end_positions)]
                assert all([x==to_be_same[0] for x in to_be_same])


            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            truncated.append(len(doc_spans))

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                        split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_positions = []
                end_positions = []
                switches = []
                answer_mask = []
                if is_training:
                    for (orig_answer_text, start_position, end_position, switch, \
                                tok_start_position, tok_end_position) in zip(\
                                original_answer_text_list, start_position_list, end_position_list, \
                                switch_list, tok_start_positions, tok_end_positions):
                        if orig_answer_text not in ['yes', 'no'] or switch == 3:
                            # For training, if our document chunk does not contain an annotation
                            # we throw it out, since there is nothing to predict.
                            doc_start = doc_span.start
                            doc_end = doc_span.start + doc_span.length - 1
                            if (tok_start_position < doc_start or
                                    tok_end_position < doc_start or
                                    tok_start_position > doc_end or tok_end_position > doc_end):
                                continue
                            doc_offset = len(query_tokens) + 2
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset
                        else:
                            start_position, end_position = 0, 0
                        start_positions.append(start_position)
                        end_positions.append(end_position)
                        switches.append(switch)
                    to_be_same = [len(start_positions), len(end_positions), len(switches)]
                    assert all([x==to_be_same[0] for x in to_be_same])

                    if sum(to_be_same) == 0:
                        if np.random.random() < 0.9:
                            continue
                        start_positions = [0]
                        end_positions = [0]
                        switches = [3]

                    if len(start_positions) > max_n_answers:
                        features_with_truncated_answers.append(len(features))
                        start_positions = start_positions[:max_n_answers]
                        end_positions = end_positions[:max_n_answers]
                        switches = switches[:max_n_answers]
                    answer_mask = [1 for _ in range(len(start_positions))]
                    for _ in range(max_n_answers-len(start_positions)):
                        start_positions.append(0)
                        end_positions.append(0)
                        switches.append(0)
                        answer_mask.append(0)

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        doc_tokens=doc_tokens,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_positions,
                        end_position=end_positions,
                        switch=switches,
                        answer_mask=answer_mask))

                unique_id += 1

    print (np.mean(truncated))
    return features, len(features_with_truncated_answers)

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index











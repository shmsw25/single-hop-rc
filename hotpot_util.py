def find_span(context, answer):
    if answer not in context or answer.strip().lower() in ['yes', 'no']:
        return None
    tokens = context.split(' ')
    offset = 0
    spans = []
    scanning = None

    for i, token in enumerate(tokens):
        while context[offset:offset+len(token)] != token:
            offset += 1
            if offset >= len(context):
                break
        if scanning is not None:
            if answer.endswith(token):
                end = offset + len(token)
                if context[scanning[-1]:end] == answer:
                    spans.append(scanning[0])
                elif len(context[scanning[-1]:end]) >= len(answer):
                    scanning = None
        if answer.startswith(token):
            if token == answer:
                spans.append(offset)
            if token != answer:
                scanning = [offset]
        offset += len(token)
        if offset >= len(context):
            break

    answers = []

    for span in spans:
        if context[span:span+len(answer)] != answer:
            print (context[span:span+len(answer)], answer)
            print (context)
            assert False
        answers.append({'text': answer, 'answer_start': span})
    return answers








import torch

def _truncate_seq(tokens, max_length: int):
    """
        cắt tokens nếu tokens cố độ dài lớn hơn max_length
    """
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        tokens.pop()

def QAtokenize(tokenizer, question: str, answers: list, max_sequence_length=512):
    """
        tạo ra các tokens đầu vào cho model QA từ question và tập hợp các answer
    """
    max_bert_input_length = 0

    question_tokenized = tokenizer.tokenize(question)
    question_length = len(question_tokenized)
    question_tokens = ["[CLS]"] + question_tokenized + ["[SEP]"]

    list_answer_tokenized = []
    for answer in answers:
        answer_tokenized = tokenizer.tokenize(answer)
        _truncate_seq(answer_tokenized, max_sequence_length - question_length - 3) 
        list_answer_tokenized.append(answer_tokenized)

        max_bert_input_length = max(max_bert_input_length, question_length + len(answer_tokenized) + 3)

    dataset_input_ids = torch.empty((len(list_answer_tokenized), max_bert_input_length), dtype=torch.long)
    dataset_attention_masks = torch.empty((len(list_answer_tokenized), max_bert_input_length), dtype=torch.long)

    for idx, answer_tokenized in enumerate(list_answer_tokenized):
        tokens = question_tokens.copy()

        for token in answer_tokenized:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_masks = [1] * len(tokens)
        while len(attention_masks) < max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)

        dataset_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
        dataset_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)

    return dataset_input_ids, dataset_attention_masks


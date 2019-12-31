# -*- coding: utf-8 -*-
import re

# Special token
SPACE_TOKEN = " [SEP] "
NUM_TOKEN = " [NUM] "

# Regular Expression
rm_space = re.compile(r"\s+")
rm_num = re.compile(r"[0-9]+")
kp_zh_en = re.compile(r"[^\u4e00-\u9fff|a-z|A-Z|0-9|\s]+")


def convert_char_token(text):
    """return a list of character"""
    text = text.lower()
    text = re.sub(kp_zh_en, "", text)
    text = re.sub(rm_num, NUM_TOKEN, text)
    text = re.sub(rm_space, SPACE_TOKEN, text)
    tokens = []
    for token in text.split():
        if token.startswith('[') and token.endswith(']'):
            tokens.append(token)
        else:
            tokens.extend(list(token))
    return tokens

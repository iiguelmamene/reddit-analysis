#!/usr/bin/env python

from __future__ import print_function
import re
import string
import argparse
import json
import sys

__author__ = ""
__email__ = ""

_EXTERNAL_PUNCTUATION = {
    ".",
    "!",
    "?",
    ":",
    ";",
    ","
}

# replace \n and \t with spaces
def replace_with_space(text):
    # use regular expressions to replace all the \n and \t with spaces
    returnStr = re.sub("[\n\t]", " ", text)
    return returnStr

# uppercase to lowercase
def replace_with_lowercase(text):
    return text.lower()

# remove urls
def remove_url(text):
    # first replace the [websiteName](urlName) with websiteName
    returnStr = re.sub(r'\[(.*)\]\((?:(?:https?|file|ftp)://)?\S+(?:\.com|\.edu|\.gov|\.org|\.net|\.us)\\*\S*\)', r'\1', text)
    # remove urls not in [websiteName](urlName) format
    returnStr = re.sub(r'(?:(?:https?|file|ftp)://)?\S+(?:\.com|\.edu|\.gov|\.org|\.net|\.us)\\*\S*', "", returnStr)
    return returnStr

# split the strings on all the spaces
def split_on_space(text):
    # use regular expressions to remove multiple spaces
    noSpaceStr = re.sub(" +", " ", text)
    # remove excess whitespace
    noSpaceStr = noSpaceStr.strip()
    # split
    returnStr = noSpaceStr.split(" ")
    return returnStr

# return the all the parsed words in a string
def create_parsed_text(list):
    # concatenate the strings
    returnStr = ""
    totalLength = len(list)
    for counter, token in enumerate(list):
        returnStr = returnStr + token
        # if not last token, add a space
        if counter < (totalLength - 1):
            returnStr = returnStr + " "
    return returnStr

# create the unigrams
def create_unigrams(list):
    # concatenate the strings
    returnStr = ""
    totalLength = len(list)
    for counter, token in enumerate(list):
        if token not in _EXTERNAL_PUNCTUATION:
            if returnStr != "":
                returnStr = returnStr + " "
            returnStr = returnStr + token
    return returnStr

# create the bigrams
def create_bigrams(list):
    # concatenate the strings
    returnStr = ""
    totalLength = len(list)
    for i in range(len(list)):
        if i < (totalLength - 1):
            if list[i] not in _EXTERNAL_PUNCTUATION and list[i + 1] not in _EXTERNAL_PUNCTUATION:
                if returnStr != "":
                    returnStr = returnStr + " "
                returnStr = returnStr + list[i] + "_" + list[i + 1]
    return returnStr

# create the trigrams
def create_trigrams(list):
    # concatenate the strings
    returnStr = ""
    totalLength = len(list)
    for i in range(len(list)):
        if i < (totalLength - 2):
            if list[i] not in _EXTERNAL_PUNCTUATION and list[i + 1] not in _EXTERNAL_PUNCTUATION and list[i + 2] not in _EXTERNAL_PUNCTUATION:
                if returnStr != "":
                    returnStr = returnStr + " "
                returnStr = returnStr + list[i] + "_" + list[i + 1] + "_" + list[i + 2]
    return returnStr

def remove_characters(text):
    # remove all the special characters and unacceptable punctuation
    newStr = re.sub(r" [^a-zA-Z1234567890\.,:;\?!]", " ", text)
    newStr = re.sub(r"[^a-zA-Z1234567890\.,:;\?!] ", " ", newStr)

    newStr = re.sub(" +", " ", newStr)
    newStr = newStr.strip()

    return newStr

def remove_external_puntuation(text):
    newStr = re.sub(r"[^a-zA-Z0123456789 `~!@\#\$%\^&\*\(\)_\-\+=,\.<>\?/\\\{\}\[\]:;\|\'\"]", "", text)

    # edge cases
    newStr = re.sub(r'(\w)(\W)$', r'\1 \2', newStr)
    newStr = re.sub(r'^(\W)(\w)', r'\1 \2', newStr)
    newStr = re.sub(r'^(\W)$', r'\1 ', newStr)

    newStr = re.sub(r'(\W)(\W)$', r'\1 \2', newStr)
    newStr = re.sub(r'^(\W)(\W)', r'\1 \2', newStr)

    # add space when external character at the end of a word
    newStr = re.sub(r'(\w)(\W) ', r'\1 \2 ', newStr)
    # add space when external character at beginning of word
    newStr = re.sub(r' (\W)(\w)', r' \1 \2', newStr)

    # use regular expressions to remove the multiple spaces
    newStr = re.sub(" +", " ", newStr)

    # multiple punctuation
    for i in range(100):
        newStr = re.sub(r' (\W)(\W)', r' \1 \2 ', newStr)
        newStr = re.sub(r'(\W)(\W) ', r' \1 \2 ', newStr)
        # use regex to remove the multiple spaces
        newStr = re.sub(" +", " ", newStr)

    newStr = re.sub(r'(\w)(\W) ', r'\1 \2 ', newStr)
    newStr = re.sub(r' (\W)(\w)', r' \1 \2', newStr)
    newStr = re.sub(" +", " ", newStr)

    return newStr

def clean_up(text):

    returnStr = replace_with_space(text)
    returnStr = remove_url(returnStr)
    returnStr = replace_with_lowercase(returnStr)
    returnStr = remove_external_puntuation(returnStr)
    returnStr = remove_characters(returnStr)
    returnStr = split_on_space(returnStr)
    parsed_text = create_parsed_text(returnStr)
    unigrams = create_unigrams(returnStr)
    bigrams = create_bigrams(returnStr)
    trigrams = create_trigrams(returnStr)

    arr = unigrams + " " + bigrams + " " + trigrams
    arr_split = arr.split()
    return arr_split

if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.stderr.write("wrong number of operands\n")
        sys.exit(1)

    # open file for processing
    with open(sys.argv[1], 'r') as f:
        for line in f:
            lineStr = json.loads(line)
            parsedStr = clean_up(lineStr['body'])
            print (str(parsedStr))

#!/usr/bin/env python

"""Clean comment text for easier parsing."""

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

# replace the new lines and tabs with spaces
def replace_with_space(text):

    # use regular expressions to replace all the newlines and tabs with spaces
    returnStr = re.sub("[\n\t]", " ", text)
    return returnStr

# replace all uppercase letters with lowercase letters
def replace_with_lowercase(text):

    return text.lower()

# remove all valid urls in the text

def remove_url(text):

    # first replace the [websiteName](urlName) with websiteName
    returnStr = re.sub(r'\[(.*)\]\((?:(?:https?|file|ftp)://)?\S+(?:\.com|\.edu|\.gov|\.org|\.net|\.us)\\*\S*\)', r'\1', text)
    # then go back and remove all remaining url's not in [websiteName](urlName) format
    returnStr = re.sub(r'(?:(?:https?|file|ftp)://)?\S+(?:\.com|\.edu|\.gov|\.org|\.net|\.us)\\*\S*', "", returnStr)
    # more complicated regex that I want to save but idk if it works as well as the other one
    # (?:(?:https?|file|ftp)://)?[a-zA-Z.@:/_\(\)?=&#-0123456789]+(?:.com|.edu|.gov|.org|.net|.us)[a-zA-Z.@:/_\(\)?=&#-0123456789]*
    return returnStr

# split the strings on all the spaces
def split_on_space(text):

    # use regex to remove the multiple spaces
    noSpaceStr = re.sub(" +", " ", text)
    # trim excess whitespace at beginning and end
    noSpaceStr = noSpaceStr.strip()
    # use the python split function to tokenize the string
    returnStr = noSpaceStr.split(" ")
    return returnStr

# return the all the parsed words in a string
def create_parsed_text(list):

    # concatenate all the strings
    returnStr = ""
    totalLength = len(list)
    for counter, token in enumerate(list):
        returnStr = returnStr + token
        # if not last token, add a space
        if counter < (totalLength-1):
            returnStr = returnStr + " "
    return returnStr

# create the unigrams in a string
def create_unigrams(list):

    # concatenate all the strings
    returnStr = ""
    totalLength = len(list)
    for counter, token in enumerate(list):
        if token not in _EXTERNAL_PUNCTUATION:
            # if not empty string, add a space
            if returnStr != "":
                returnStr = returnStr + " "
            returnStr = returnStr + token
    return returnStr

# create the unigrams in a string
def create_bigrams(list):

    # concatenate all the strings
    returnStr = ""
    totalLength = len(list)
    for i in range(len(list)):
        # only go through the second to last word
        if i < (totalLength-1):
            if list[i] not in _EXTERNAL_PUNCTUATION and list[i+1] not in _EXTERNAL_PUNCTUATION:
                # only add space if not first bigram in string
                if returnStr != "":
                    returnStr = returnStr + " "
                returnStr = returnStr + list[i] + "_" + list[i+1]
    return returnStr

# create the trigrams in a string
def create_trigrams(list):

    # concatenate all the strings
    returnStr = ""
    totalLength = len(list)
    for i in range(len(list)):
        # only go through the third to last word
        if i < (totalLength-2):
            if list[i] not in _EXTERNAL_PUNCTUATION and list[i+1] not in _EXTERNAL_PUNCTUATION and list[i+2] not in _EXTERNAL_PUNCTUATION:
                # only add space if not first trigram in string
                if returnStr != "":
                    returnStr = returnStr + " "
                returnStr = returnStr + list[i] + "_" + list[i+1] + "_" + list[i+2]
    return returnStr

def test_remove(text):

    # remove all the special characters and unacceptable punctuation
    newStr = re.sub(r" [^a-zA-Z1234567890\.,:;\?!]", " ", text)
    newStr = re.sub(r"[^a-zA-Z1234567890\.,:;\?!] ", " ", newStr)

    newStr = re.sub(" +", " ", newStr)
    newStr = newStr.strip()

    return newStr

def test_external(text):

    # remove the wild shit
    newStr = re.sub(r"[^a-zA-Z0123456789 `~!@\#\$%\^&\*\(\)_\-\+=,\.<>\?/\\\{\}\[\]:;\|\'\"]", "", text)

    # edge cases of beginning or end of string
    newStr = re.sub(r'(\w)(\W)$', r'\1 \2', newStr)
    newStr = re.sub(r'^(\W)(\w)', r'\1 \2', newStr)
    newStr = re.sub(r'^(\W)$', r'\1 ', newStr)

    newStr = re.sub(r'(\W)(\W)$', r'\1 \2', newStr)
    newStr = re.sub(r'^(\W)(\W)', r'\1 \2', newStr)

    # add space when external character at the end of a word
    newStr = re.sub(r'(\w)(\W) ', r'\1 \2 ', newStr)
    # add space when external character at beginning of word
    newStr = re.sub(r' (\W)(\w)', r' \1 \2', newStr)


    # use regex to remove the multiple spaces
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

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    returnStr = replace_with_space(text)
    returnStr = remove_url(returnStr)
    returnStr = replace_with_lowercase(returnStr)
    returnStr = test_external(returnStr)
    returnStr = test_remove(returnStr)
    returnStr = split_on_space(returnStr)
    parsed_text = create_parsed_text(returnStr)
    unigrams = create_unigrams(returnStr)
    bigrams = create_bigrams(returnStr)
    trigrams = create_trigrams(returnStr)

    #return [parsed_text, unigrams, bigrams, trigrams]
    arr1 = unigrams + " " + bigrams + " " + trigrams
    arr2 = arr1.split()
    return arr2

if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python parse-text.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # check the number of arguments
    if len(sys.argv) != 2:
        sys.stderr.write("wrong number of operands\n")
        sys.exit(1)

    # open the test file for processing
    with open(sys.argv[1], 'r') as f:
        for line in f:
            lineStr = json.loads(line)
            parsedStr = sanitize(lineStr['body'])
            print (str(parsedStr))

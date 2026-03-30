import os
import pandas
import numpy
import math
import unicodedata
from collections import defaultdict
import heapq
import regex as re
from collections import Counter, defaultdict
import time

hinditex= r"C:\Users\Shivam Kumar\Downloads\wmt-news-crawl-hi.txt"

nonhinditokens = {0: "<UNK>", 1: "<s>", 2: "</s>"}
vocab= {} 
idnum_to_token = {}

#part 1
#preprocessing!!!!!!!!!

#nkfc
def preprocessor(hinditext):
    processed_sentencesall = []

    with open(hinditext, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace('\u200c', '').replace('\u200d', '')
            nfkctext = unicodedata.normalize("NFKC", line)

            spacedtext = []
            for chara in nfkctext:
                if unicodedata.category(chara).startswith("P"):
                    spacedtext.append(" ")
                    spacedtext.append(chara)
                    spacedtext.append(" ")
                else:
                    spacedtext.append(chara)

            result = "".join(spacedtext)
            result = " ".join(result.split())

            processed_sentencesall.append(result)

    return processed_sentencesall


#part 2
#!!!!!! tariner
#
def train(hinditext,vocabtargetsize):
    #core
    #globalwords never changes
    #it just maintains frequency of words we had at very start 
    #split changes
    #in iteration till i reach target
    #
    global vocab, idnum_to_token
    processedlines = preprocessor(hinditext)
    hinditext=processedlines
    globalword_counts = Counter()
    #this creates a unique word vs frequency map
    #will be global
    for line in hinditext:
        globalword_counts.update(line.strip().split())

    #does ## for non start 
    #dict to hold all splits
    splits = {}
    #this holds word splits like h,##e,##l
    #for all unique words
    for word in globalword_counts:
        #take matra with a leter itself
        chars = re.findall(r"\X", word)
        #if we want to split even matra
        #!!!uncomment this
        #chars = list(word)
        # splits[word] = [chars[0]] + ["##" + c for c in chars[1:]]
        result=[]
        for i in range(len(chars)):
            if i == 0:
                result.append(chars[i])
            else:
                result.append("##" + chars[i])
        splits[word]=result
    
    vocab_set = set()
    #looks like [a,##b,##c,]
    for sentence in splits.values():
        for token in sentence:
            vocab_set.add(token)

    while len(vocab_set)<(vocabtargetsize-len(nonhinditokens)):
        adjpair_count = defaultdict(int)
        token_count = defaultdict(int)
        #i have ["w", "##a", "##s"]
        #so it does wa+=1 as+=1
        for word, count in globalword_counts.items():
            split = splits[word]
            for i in range(len(split) - 1):
                adjpair_count[(split[i], split[i+1])] += count
            for token in split:
                token_count[token] += count

        if not adjpair_count:
            break

        #likelihood Score = freq(pair) / (freq(a) * freq(b))
        best_score = -1
        candidates = []

        for pair, freq in adjpair_count.items():
            score = freq / (token_count[pair[0]] * token_count[pair[1]])
            if score > best_score:
                best_score = score
                candidates = [pair]
            elif score == best_score:
                candidates.append(pair)

        #tie break: pair that appears first in element-wise lexicographic order
        replacement = min(candidates) 
        
        #merge the pair
        a, b = replacement
        if len(b) >= 2 and b[0] == "#" and b[1] == "#":
            new_token = a
            i = 2
            while i < len(b):
                new_token = new_token + b[i]
                i += 1
        else:
            new_token = a + b

        vocab_set.add(new_token)

        #now for all i hve to update by the merrge
        #liike h,##e to he
        for word in splits:
            #splits->has word vs split of that word
            split = splits[word]
            i = 0
            newsplit= []
            while i < len(split):
                if i < len(split) - 1 and split[i] == replacement[0] and split[i+1] == replacement[1]:
                    newsplit.append(new_token)
                    i += 2
                else:
                    newsplit.append(split[i])
                    i += 1
            splits[word]=newsplit

    #arranging in lexico order
    sortedvocab= sorted(list(vocab_set))
    idnum_to_token = {0: "<UNK>", 1: "<s>", 2: "</s>"}
    x=3
    for i, token in enumerate(sortedvocab):
        idnum_to_token[x+i] = token
        #inverting the vocab to word to id
    vocab = {v: k for k, v in idnum_to_token.items()}

#part 3
#encoder!!
def encoder_tokens(hinditext,vocab):

    processedlines = preprocessor(hinditext)
    hinditext=processedlines
    #this is a list of lists
    #for each sent hjolds tokens for that sent
    listoftokenlist=[]

    #i have processed hindi lines
    for sentence in hinditext:
        tokenlist=["<s>"]
        #entered one single line's words
        for word in sentence.split():
            #entered tokens of a word like h,e,q
            graphemest = re.findall(r"\X", word)
            start = 0
            #main logic basically try from end
            #if i can merge that fine break and merge
            #decrese on the end
            while start<len(graphemest):
                end=len(graphemest)
                mergetoken=None
                while start < end:
                    #substr from start to sec ptr
                    substr = ""
                    i = start
                    while i < end:
                        substr = substr + graphemest[i]
                        i += 1
                    if start > 0:
                        #coz its not start
                        substr = "##" + substr
                    
                    if substr in vocab:
                        mergetoken = substr
                        break
                    end-=1
                
                if mergetoken is None:
                    tokenlist.append("<UNK>")
                    break
                    
                tokenlist.append(mergetoken)
                start = end
        tokenlist.append("</s>")
        listoftokenlist.append(tokenlist)
    return listoftokenlist

def encoder_tokensid(hinditext,vocab):
    processedlines = preprocessor(hinditext)
    hinditext=processedlines
    #this is a list of lists
    #for each sent hjolds tokens for that sent
    listoftokenlist=[]

    #i have processed hindi lines
    for sentence in processedlines:
        #start with <s> ID
        tokenidlist = [vocab.get("<s>", 0)]

        for word in sentence.split():
            graphemes = re.findall(r"\X", word)
            start = 0

            while start < len(graphemes):
                end = len(graphemes)
                mergetoken = None

                while start < end:
                    substr = ""
                    i = start
                    while i < end:
                        substr = substr + graphemes[i]
                        i += 1
                    if start > 0:
                        substr = "##" + substr

                    if substr in vocab:
                        mergetoken = substr
                        break
                    end -= 1

                if mergetoken is None:
                    tokenidlist.append(vocab.get("<UNK>", 0))
                    break

                tokenidlist.append(vocab[mergetoken])
                start = end

        #eos token
        tokenidlist.append(vocab.get("</s>", 0))
        listoftokenlist.append(tokenidlist)

    return listoftokenlist

def decoder(tokenid_input_file, vocab_file,output_file):
    idnum_to_token = {}

    #firstly each token in vocab is assigned a id
    #id is simply index
    with open(vocab_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            idnum_to_token[idx] = line.strip()

    #i have a line like 1 3 5 9 19
    #i split on base of space 
    #now just make a list of it
    tokenid_sentences = []

    with open(tokenid_input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tokenid_sentences.append(list(map(int, line.strip().split())))

    decoded_sentences = []

    #going for one sent in a list of lists
    for tokenidlist in tokenid_sentences:
        sentence = ""
        #get the token from the map we made
        for tid in tokenidlist:
            token = idnum_to_token.get(tid, "<UNK>")

            if token == "<s>" or token == "</s>":
                continue

            #if its not a start append it to lst one
            if len(token) >= 2 and token[0] == "#" and token[1] == "#":
                sentence += token[2:]
                #else make a space then add it
            else:
                if sentence:
                    sentence += " "
                sentence += token

        #removing extra spaces
        sentence = " ".join(sentence.split())
        decoded_sentences.append(sentence)

    #write line by ine
    with open(output_file, "w", encoding="utf-8") as f:
        for line in decoded_sentences:
            f.write(line + "\n")


if __name__ == "__main__":

    #main loop
    input_file = hinditex

    preprocessed_file = "preprocessed.txt"
    vocab_file = "vocabulary.txt"
    tokens_file = "tokens.txt"
    tokenids_file = "token_ids.txt"
    decoded_file = "decoded.txt"

    vocab_size = 10000


    #preprocessor
    print("part 1 preprocessing the raw file")

    processedlines = preprocessor(input_file)

    with open(preprocessed_file, "w", encoding="utf-8") as f:
        for line in processedlines:
            f.write(line + "\n")


    #train 
    print("training vocab target is 10000 words")

    train(input_file,vocab_size)

    with open(vocab_file, "w", encoding="utf-8") as f:
        for idx in sorted(idnum_to_token):
            f.write(idnum_to_token[idx] + "\n")


    #takes raw file preprocesss it then gives tokens
    print("Step 3: Encoding input file into tokens...")

    token_lists = encoder_tokens(input_file, vocab)

    with open(tokens_file, "w", encoding="utf-8") as f:
        for sent in token_lists:
            f.write(" ".join(sent) + "\n")


    #same but with id instead of token
    print("Step 4: Encoding input file into token IDs...")

    tokenid_lists = encoder_tokensid(input_file, vocab)

    with open(tokenids_file, "w", encoding="utf-8") as f:
        for sent in tokenid_lists:
            f.write(" ".join(map(str, sent)) + "\n")


    #decode with part 3 4
    print("Step 5: Decoding token IDs back to text...")

    decoder(tokenid_input_file=tokenids_file,vocab_file=vocab_file,output_file=decoded_file)

    print("done.")

import nltk
import sys
import string
import os
import math
import spacy
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
lemmatizer = WordNetLemmatizer()
FILE_MATCHES = 1
SENTENCE_MATCHES = 1
vocab_set=set()
en_nlp = spacy.load('en')

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])

    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    
    vocab=list(vocab_set)

    file_idfs = compute_idfs(file_words)

    file_vectors={
        filename: vectorize(file_words[filename],file_idfs,vocab)
        for filename in files
    }

    # Prompt user for query
    query_statement=input("Query: ")
    query = set(tokenize(query_statement))
    
    # for word in query:
    #     print(synonyms(word))
    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, vocab, file_vectors, n=FILE_MATCHES)
    # print(filenames)
    
    # Extract sentences from top files
    query_roots=parse_tree_roots(query_statement)
    ptr_feature=dict()
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                roots=parse_tree_roots(sentence)
                b=0
                for r in query_roots:
                    if (r in roots):
                        b=1
                ptr_feature[sentence]=b
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, ptr_feature, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dic=dict()
    for fn in os.listdir(directory):
        fa=os.path.join(directory,fn)
        file=open(fa)
        s=file.read()
        file.close
        dic[fn]=s

    return dic    


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    r = [x.lower() for x in nltk.tokenize.word_tokenize(document)]
    words=[]
    en_stops = set(nltk.corpus.stopwords.words('english'))
    punc = string.punctuation
    for x in r:
        if x not in punc and x not in en_stops and len(x)>1 :
            x=lemmatizer.lemmatize(x)
            words.append(x)
            vocab_set.add(x)

    return words     


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    cnt=dict()
    for doc in documents:
        s=set()
        for word in documents[doc]:
            s.add(word)
        for word in s:        
            cnt[word]=cnt.get(word,0)+1
    idf=dict()
    l=len(documents)
    for word in cnt:
        idf[word]=math.log((l+1)/(cnt[word]+1))        
    
    return idf

def vectorize(tokens,file_idfs,vocab):
    """
    Given a list of tokenized words and dictionary mapping words to idf  
    return a numpy array of TF values words indexed according to vocab set
    """
    
    Q=np.zeros(len(vocab))
    counter=Counter(tokens)
    word_count=len(tokens)
    for token in np.unique(tokens):
        tf=counter[token]/word_count
        idf=file_idfs.get(token,0)
        Q[vocab.index(token)]=tf*idf
    
    return Q
    
def parse_tree_roots(sentence):
    """
    Given a sentence, returns a list of roots/subroots of dependency 
    parse tree of sentence 
    """
    doc=en_nlp(sentence)
    roots=[]
    for sent in doc.sents:
        roots += [lemmatizer.lemmatize(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]
        roots += [lemmatizer.lemmatize(sent.root.head.text.lower())]

    return set(roots)    

def top_files(query, files, idfs, vocab, file_vectors, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    dic={}
    query=list(query)
    Q=vectorize(query,idfs,vocab)

    for file in files:
        d=Counter(files[file])
        P=file_vectors[file]
        dic[file]=np.dot(P,Q)/(np.linalg.norm(P)*np.linalg.norm(Q))
        for word in query:
            val=0
            for syn in synonyms(word):
                new_val=d.get(syn,0) * idfs.get(syn,0)
                if(word!=syn):
                    new_val*=0.3
                val=max(val,new_val)
            dic[file] = dic[file] + val

    lst = sorted(dic.items(), key =lambda kv:(kv[1], kv[0]),reverse=True)[0:n] 
    return [x[0] for x in lst]


def top_sentences(query, sentences, idfs, ptr_feature, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    dic1={}
    for sentence in sentences:
        d={}
        for word in sentences[sentence]:
            d[word]=1
        for word in query:
            val=0
            for syn in synonyms(word):
                new_val=(d.get(syn,0) * idfs.get(syn,0))
                if(word!=syn):
                    new_val=0.3*val
                val=max(val,new_val)
            dic1[sentence]  = dic1.get(sentence,0) + val

    dic2={}
    for sentence in sentences:
        d=Counter(sentences[sentence])
        for query_word in query:
            val=0
            for syn in synonyms(query_word):
                new_val=d.get(syn,0)
                if(query_word!=syn):
                    new_val*=0.3
                val=max(val,new_val)    
            dic2[sentence] = dic2.get(sentence,0) + val
        dic2[sentence] = dic2.get(sentence,0) / len(sentences[sentence])
    
    dic3={}
    for sentence in sentences:
        dic3[sentence]=(ptr_feature[sentence] * dic1.get(sentence,0),dic2.get(sentence,0))
    
    lst = sorted(dic3.items(), key =lambda kv:(kv[1], kv[0]),reverse=True)[0:n]
    # for x in lst:
    #     print(x[1])
    return [x[0] for x in lst]

def synonyms(word):
    """
    Given a word, returns a set of its synonyms 
    """
    syn_list=set()
    for syn in wordnet.synsets(word):
        for name in syn.lemma_names():
            syn_list.add(name)

    syn_list.add(word)
    return syn_list  
      
if __name__ == "__main__":
    main()

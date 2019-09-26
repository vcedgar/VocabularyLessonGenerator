import spacy
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES, English
import pandas

from Prototype.Vocabulary.Word import Word
from Prototype.Vocabulary.Sentence import Sentence

class Document:
    def __init__(self, text, maxChunkSize, maskingToken):
        #Loading stuff:
        nlp = spacy.load('en_core_web_md')

        self.maskToken = maskingToken
        french = English()
        sentencizer = french.create_pipe("sentencizer")

        # sentences = french(text).sents
        nlp.add_pipe(sentencizer, before="parser")
        spacyDoc = nlp(text)
        sentences = spacyDoc.sents
        cls = Word('[CLS]', 'na', False, ['[CLS]'], maskingToken)
        eos = Word('[SEP]', 'na', False, ['[SEP]'], maskingToken)
        self.document = []
        self.noMasked = True

        bertTokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
        model = BertForMaskedLM.from_pretrained('bert-large-cased-whole-word-masking')

        # Go through each sentence
        for sent in sentences:
            # Tokenize the sentence and go through each word
            sentenceTokens = [cls.text]
            sentenceWords = [cls]
            sentenceUnmaskedTokens = [cls.text]
            for word in sent:
                newWord = self.makeWord(word)
                # add the word's sub word tokens to the chunk
                sentenceTokens.extend(newWord.maskedTokens)
                sentenceWords.append(newWord)
                sentenceUnmaskedTokens.extend(newWord.actualTokens)
            sentenceTokens.append(eos.text)
            sentenceUnmaskedTokens.append(eos.text)
            sentenceWords.append(eos)
            newSent = Sentence(sentenceWords, sentenceTokens, sentenceUnmaskedTokens, self.maskToken, bertTokenizer,
                               model)

            # print(maskedChunk)
            self.document.append(newSent)

    def makeWord(self, word):
        bertTokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
        knownWords = pandas.read_csv("./Lexique382/EnglishVocab.tsv", sep='\t').lemme.to_list()

        # get the lemma for the current word
        lemma = word.lemma_
        isMasked = False
        # determine if this word should be masked
        if word.is_alpha and (word.ent_iob == 0 or word.ent_iob == 2) and lemma not in knownWords:
            isMasked = True
        # tokenize according to BERT
        subwordTokens = bertTokenizer.wordpiece_tokenizer.tokenize(word.text)
        # create the new word
        return Word(word.text, lemma, isMasked, subwordTokens, self.maskToken)

import pandas
import spacy


nlp = spacy.load('fr_core_news_sm')
vocabulary = pandas.read_csv("./Lexique382/lexique3_words_90_percentile.tsv", sep='\t').lemme.to_list()

f=open("./Books/Year2025-ToChap2.txt", "r")
year2025 = nlp(f.read())

spacy.tokens.Token.set_extension("isUnknown", default=False)

def markUnknowns(doc, vocab):
    countU = 0
    countA = 0
    for token in doc:
        countA+=1
        if token.lemma_ not in vocab and token.is_alpha:
            token._.isUnknown = True
            countU +=1
    doc._.unknownWordsCount = countU
    doc._.totalWordsCount = countA

def printKnowns(doc):
    for token in doc:
        if not token._.isUnknown:
            print(token.text, end=" ")


spacy.tokens.Doc.set_extension("unknownWordsCount", default=-1)
spacy.tokens.Doc.set_extension("totalWordsCount", default=-1)
spacy.tokens.Doc.set_extension("printKnowns", method=printKnowns)
spacy.tokens.Doc.set_extension("markUnknowns", method=markUnknowns)

year2025._.markUnknowns(vocabulary)
print("Number Unknown: ", year2025._.unknownWordsCount)
print("Number Total: ", year2025._.totalWordsCount)
print("Percent Unkown: ", (year2025._.unknownWordsCount/year2025._.totalWordsCount)*100)

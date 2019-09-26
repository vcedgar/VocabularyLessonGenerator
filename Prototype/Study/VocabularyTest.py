import pandas
import math

#*****************************REDO as BinSearch********************************************

def runQuartileTest(vocab):
    avg = 0
    total = 0
    wordsUnderstood = len(vocab)
    uInput = ""

    for word in vocab:
        total += 1
        uInput = input(str(total) + "Do you know this word: " + word)
        if uInput == "y":
            avg = (avg * (total - 1) + 1)/total
        else:
            avg = (avg * (total - 1)) / total

        if avg >.98:
            wordsUnderstood = total
    if avg < .5:
        wordsUnderstood = 0
    return wordsUnderstood


def makeQuartileTest(vocab):
    count = vocab.count()
    wordsInPercent = math.ceil(count/100) - 1
    index = 0
    testList = []
    while index <= count - wordsInPercent:
        percent = vocab.take(list(range(index, index+wordsInPercent)))
        testList.append(percent.sample(1).tolist()[0])
        index = index + wordsInPercent
    print(wordsInPercent)
    return testList

def vocabTest(vocab):
    count = vocab.count()
    wordsInQuartile = math.ceil(count/4) - 1
    index = 0
    quartiles = []
    while index <= count - wordsInQuartile:
        quartile = vocab.take(list(range(index, index+wordsInQuartile)))
        quartileTest = makeQuartileTest(quartile)
        testResult = runQuartileTest(quartileTest)
        index += wordsInQuartile
        quartiles.append({"quartile": quartile, "wordsToLearn": wordsInQuartile - wordsInQuartile*(testResult//100)})
    firstQuartile = 0
    lastQuartile = 3
    for i in range(2, 4):
        if quartiles[i]["wordsToLearn"] == wordsInQuartile:
            lastQuartile = i-1
    for i in range(0, 2):
        if quartiles[i]["wordsToLearn"] == 0:
            firstQuartile = i + 1
    print(firstQuartile, " ", lastQuartile)
    start = firstQuartile * wordsInQuartile + quartiles[firstQuartile]["wordsToLearn"]
    end = 0
    if(firstQuartile == lastQuartile):
        end = lastQuartile * wordsInQuartile
    else:
        end = lastQuartile * wordsInQuartile + quartiles[lastQuartile]["wordsToLearn"]
    print("Start: ", start, "End: ", end)
    f = open("wordIndex.txt", "w")
    f.write("Start: " + str(start) + " End: " + str(end))
    f.close()
    return vocab.take(list(range(start, end)))

# Function used to create the Lexique382/lexique3_words_90_percentile.tsv file from the Lexique382/Lexique382.tsv file

def getLexique90Percentile():
    lexique = pandas.read_csv("./Lexique382/Lexique382.tsv", sep='\t')

    n = 100000
    mostfreqlemma = lexique.drop(["ortho", "phon", "cgram", "genre", "nombre", "freqlemfilms2", "freqfilms2",
                  "freqlivres", "infover", "nbhomogr", "nbhomoph", "islem", "nblettres", "nbphons", "cvcv", "p_cvcv",
                  "voisorth", "voisphon", "puorth", "puphon", "syll", "nbsyll", "cv-cv", "orthrenv", "phonrenv",
                  "orthosyll", "cgramortho", "deflem", "defobs", "old20", "pld20", "morphoder", "nbmorph"],
                                    1).drop_duplicates().nlargest(n, "freqlemlivres")
    sumfll = mostfreqlemma.freqlemlivres.cumsum()
    sumfllpercent = sumfll.apply((lambda x: x/10000))
    mostfreqlemmadata = mostfreqlemma.join(sumfll, rsuffix="_cumsum").join(sumfllpercent, rsuffix="_cumsumpercent")

    wordsneeded = mostfreqlemmadata[mostfreqlemmadata["freqlemlivres_cumsumpercent"] <= 90]
    wordsneeded.to_csv(path_or_buf="./Lexique382/lexique3_words_90_percentile.tsv", sep="\t")


getLexique90Percentile()
vocabulary = pandas.read_csv("./Lexique382/lexique3_words_90_percentile.tsv", sep='\t')
maxVocab = 5000
vocabulary.head(maxVocab).to_csv(path_or_buf="./Lexique382/myVocab.tsv", sep="\t")
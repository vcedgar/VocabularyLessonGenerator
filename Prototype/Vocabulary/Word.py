class Word:
    def __init__(self, text, lemma, isMasked, tokens, maskingToken):
        self.text = text
        self.lemma = lemma
        self.isMasked = isMasked
        self.actualTokens = tokens
        self.tokenCount = len(self.actualTokens)
        self.correctPercent = 0
        self.maskedTokens = []
        self.predictedTokens = []
        if self.isMasked:
            for token in self.actualTokens:
                if not token.isalpha:
                    self.maskedTokens.append(token)
                else:
                    self.maskedTokens.append(maskingToken)
        else:
            self.maskedTokens = self.actualTokens

    def updatePredictedTokens(self, pts):
        self.predictedTokens = pts
        correctCount = 0
        for i in range(0, self.tokenCount):
            if self.actualTokens[i] in self.predictedTokens[i]:
                correctCount += 1
        if self.tokenCount == 0:
            self.correctPercent = 0
        elif self.isMasked:
            self.correctPercent = correctCount/self.tokenCount

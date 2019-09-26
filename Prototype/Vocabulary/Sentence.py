import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM


class Sentence:
    def __init__(self, ws, toks, actualToks, maskingToken, bertTokenizer, model):
        self.words = ws
        self.tokens = toks
        self.actualTokens = actualToks
        self.predictedTokens = []
        self.maskedIndices = []
        self.tensors = []

        self.numbMasked = 0
        self.numMaskedPredicted = 0

        i = 0
        for tok in self.tokens:
            if tok == maskingToken:
                self.maskedIndices.append(i)
            i += 1
            self.predictedTokens.append(tok)

        self.tensors = torch.tensor(bertTokenizer.convert_tokens_to_ids(self.tokens)).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            outputs = model(self.tensors)
            predictions = outputs[0][0]

        for index in self.maskedIndices:
            predictedIndices = torch.topk(predictions[index], 30).indices
            predictedTokens = bertTokenizer.convert_ids_to_tokens(predictedIndices.tolist(), skip_special_tokens=True)
            self.predictedTokens[index] = predictedTokens

        predictionIndex = 0
        for word in self.words:
            lengthOfWord = len(word.maskedTokens)
            word.updatePredictedTokens(self.predictedTokens[predictionIndex: (predictionIndex + lengthOfWord)])
            predictionIndex += lengthOfWord

        self.numbMasked = len(self.maskedIndices)
        for word in self.words:
            if word.isMasked and word.correctPercent == 1:
                self.numMaskedPredicted += 1

        print(self.numMaskedPredicted)

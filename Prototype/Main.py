from Prototype.Vocabulary.Document import Document
from pytorch_transformers import BertConfig
import re

with open("./Books/text.txt", "r") as file:
    year2025Text = file.read()

year2025Doc = Document(year2025Text, 10, "[MASK]")


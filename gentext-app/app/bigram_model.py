import random
from collections import defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bigrams = defaultdict(list)
        self._build(corpus)

    def _build(self, corpus):
        for text in corpus:
            words = text.lower().split()
            for i in range(len(words) - 1):
                self.bigrams[words[i]].append(words[i + 1])

    def generate_text(self, start_word, length):
        if length <= 0:
            return ""

        current = start_word.lower()
        output = [current]

        for _ in range(length - 1):
            next_words = self.bigrams.get(current)
            if not next_words:
                break

            recent = output[-3:]
            valid_next_words = [word for word in next_words if word not in recent]
            if not valid_next_words:
                break

            current = random.choice(valid_next_words)
            output.append(current)

        return " ".join(output)

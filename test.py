
from collections import Counter
def count_words(string):
    freq = Counter(string)
    return (freq[' '] + 1)

texxt = """model: The model parameter specifies the pre-trained model to be used for the given task.
			In your example, the model is set to "facebook/bart-large-cnn", which refers to the BART
			(Bidirectional and AutoRegressive Transformers) model pre-trained by Facebook AI.
			This particular variant, "bart-large-cnn," is trained on a large corpus of data and is known for its good performance on summarization tasks.
"""


x = count_words(texxt)

print(x)

print(int(x * 0.5)+1)

print(int(x * 0.25))



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline

# lead aspects based pre-trained model 
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

text = "Our birthday party diner had great experience at the solidad restaurant, food was delicious as usual," \
"but the service was ok, photo service was excellent."

# ABSA of food
aspect = "food"
input = absa_tokenizer(f"[CLS] {text} [SEP] {aspect} [SEP]", return_tensors="pt")
outputs = absa_model(**input)
probs = F.softmax(outputs.logits, dim=1)
probs = probs.detach().numpy()[0]

print(f"sentense : {text}")
print("*****")
print(f"Sentiment of {aspect}: ")
for prob, label in zip(probs, ["neg", "neu", "pos"]):
    print (f"Label {label}: {prob}")
print("*****")

aspect = "service"
input = absa_tokenizer(f"[CLS] {text} [SEP] {aspect} [SEP]", return_tensors="pt")
outputs = absa_model(**input)
probs = F.softmax(outputs.logits, dim=1)
probs = probs.detach().numpy()[0]

print(f"Sentiment of {aspect}: ")
for prob, label in zip(probs, ["neg", "neu", "pos"]):
    print (f"Label {label}: {prob}")    
print("*****")
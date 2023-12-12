
# import pre-trained tokenizer and model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline


class ABSA():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.classifier = pipeline("text-classification", model=self.absa_model, tokenizer=self.tokenizer)

    def getAspectScoring(self, aspect, text):
        input = self.tokenizer(f"[CLS] {text} [SEP] {aspect} [SEP]", return_tensors="pt")
        outputs = self.absa_model(**input)
        probs = F.softmax(outputs.logits, dim=1)
        probs = probs.detach().numpy()[0]       
     
        print(f"sentense : {text}")
        print("*****")

        print(f"Sentiment of {aspect}: ")
        for prob, label in zip(probs, ["neg", "neu", "pos"]):
            print (f"Label {label}: {prob}")
        print("*****")

    def getMAspects(self, aspects, text):
        for aspect in aspects:
            print(aspect, self.classifier(text, text_pair=aspect))

if __name__=="__main__":
    absa = ABSA()
    text = "Our birthday party diner had great experience at the Zyper restaurant, food was delicious as usual," \
"but the service was average, photo service was exquizit."    
    aspect = "food"
    absa.getAspectScoring(aspect=aspect, text=text) 

    # multiple aspects in list
    absa.getMAspects(aspects=["food", "service"], text=text)



# import pre-trained tokenizer and model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline


class ABSA():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1",truncation=True, max_length=512)
        self.absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.classifier = pipeline("text-classification", model=self.absa_model, tokenizer=self.tokenizer)
        self.zero_pipe = pipeline("zero-shot-classification",model=self.absa_model, tokenizer=self.tokenizer)
   
    def getAspectScoring(self, aspect, text):
        input = self.tokenizer(f"[CLS] {text} [SEP] {aspect} [SEP]", return_tensors="pt")
        outputs = self.absa_model(**input)
        probs = F.softmax(outputs.logits, dim=1)
        probs = probs.detach().numpy()[0]       
     
        print(f"sentense : {text}")
        print("*****")

        print(f"Sentiment of {aspect}: ")
        for prob, label in zip(probs, ["negative", "neutral", "positive"]):
            print (f"Label {label}: {prob}")

    def getMAspects(self, aspects, text):
        print("*****")
        for aspect in aspects:
            print(aspect, self.classifier(text, text_pair=aspect))

    def getMZAspects(self, aspects, text):
        print("*****")
        print(self.zero_pipe(text, candidate_labels=aspects))

if __name__=="__main__":
    absa = ABSA()
    text = "While we had some early issues with the Google Pixel 8 Pro mostly with its cameras further testing and software updates from Google meant we eventually came away thinking quite highly of the Pixel 8 Pro. It's got a lot going for it, from its slick design and easy-to-use Android 14 software to its rear camera setup, which can take some genuinely beautiful images." \
"Google has sprinkled a variety of new AI features throughout the phone too, including a tool that uses generative AI to create wallpapers for the phone, and a camera feature that allows you to combine different faces from a series of burst images of a group of friends to make sure that everyone looks their best." \
"But one of the best updates is Google's commitment to providing seven years of security updates to the Pixel 8 series, meaning this phone will still be safe to use in at least 2030." \
"While we haven't been as impressed with this year's 8 Pro as we were with the 7 Pro, it's still a great phone to consider, especially if you're keen to hold on to your phone for longer."    
    aspect = "camera"
    absa.getAspectScoring(aspect=aspect, text=text) 

    # multiple aspects in list
    absa.getMAspects(aspects=["camera", "security", "AI", "design"], text=text)

    # zero shor classification
    absa.getMZAspects(aspects=["camera", "security", "AI", "design"], text=text)

    #witch google pixel to choose?
    pixel7a_review = "The Pixel 7A (8/10, WIRED Recommends) is our top choice for most people because it has pretty much everything you need, setting a new standard for mid-range smartphones. (Though it is a little pricier than the Pixel 6A from 2022.) This iteration includes wireless charging, which is really uncommon for phones under $500. You also get a 90-Hz screen refresh rate, which makes the onscreen interactions look and feel more fluid, as there are more frames per second than before. " \
"Its design and build are similar to its predecessor, though Google claims the aluminum mid-frame is slightly more durable. There's Gorilla Glass 3 on the front 6.1-inch screen, which isn't as scratch-resistant as the glass on pricier Pixels, but at least the rear is a plastic composite that won't crack. This model comes in Charcoal, Sea, and Snow, but a Google Store-exclusive Coral has caught my eye." \
"You won't run into any problems with performance. It's powered by Google's Tensor G2 chipset, which is the same one that's inside the Pixel 7 series. It's plenty fast for everyday tasks and most mobile gaming, and it also allows for a few new software tricks, like Photo Unblur, which can use machine-learning algorithms to unblur people's faces in those late-night pics. (It actually did this really well when I tried to capture a photo of a bride and groom on the dance floor!)"\
"Speaking of the camera system, a 64-megapixel sensor leads the pack joined by a 13-megapixel ultrawide, and while these are not as good as the cameras in the Pixel 7, the results are nearly imperceptible (you have to look really close). It's easily the best camera phone for the money, whether you're using it in low light to capture the stars or shooting a landscape on a sunny day. " \
"One reason we recommend Pixel phones is that they include many great software features that are genuinely helpful. We've collected most of them below, but my favorites are Assistant Voice Typing for much faster and more accurate voice transcriptions using the built-in keyboard; Now Playing to find out what music is playing around me; and Call Screen, which has pretty much stopped all spam calls coming my way. You'll also get five years of security updates, so your Pixel 7A will be supported for quite a while. Unfortunately, Google only promises three OS upgrades." \
"The only things that are iffy on the Pixel 7A are the fingerprint sensor, which isn't as snappy or reliable as I'd like, and the battery life. The 4,385-mAh cell can take you through a full day of average use, but on busy days you will most likely need to top up before the sun goes down. There's also no microSD card slot or headphone jack, so you'll have to look elsewhere if you want a phone with those features. "

    pixel8_review = "If you want the latest and greatest, then go for Google's Pixel 8 and Pixel 8 Pro (7/10, WIRED Recommends). These flagships are slightly more expensive this year, but they're the only Pixels to receive a software commitment of seven years. That's right, Google is finally promising seven years of security and Android OS upgrades to these phones, outclassing all its Android peers. That even includes stocking up on spare parts for repairs to the hardware. " \
"The 8 series has the brightest OLED displays ever on a Pixel, which means no need to squint when staring at the screen on a sunny day. They have a more rounded design that makes 'em comfy to hold, though this is more evident on the 6.2-inch Pixel 8, which is smaller than its predecessor. It has a glossy glass back, whereas the 6.7-inch Pixel 8 Pro has a matte soft-touch glass back. Both have 120-Hz screen refresh rates, but only the Pro model can adjust this from 1 Hz to 120 Hz depending on what's on the screen, which is more battery-efficient. " \
"These phones now have secure Face Unlock, making them the first since the Pixel 4 to have a biometric authentication tool like Apple's Face ID. You can use your face or the in-display fingerprint scanner to access sensitive apps like your banking app. It just doesn't work well in darkness, so you'll still have to rely on your thumbs. " \
"Inside is Google's Tensor G3 processor, which hasn't given me any trouble with any of the apps or games I threw at it. It notably powers new smart imaging features, like Magic Editor, which lets you move subjects around in your photos and even change the type of sky; Best Take, for fixing people's faces in case they blinked; and Audio Magic Eraser, to remove unwanted sounds like a fire truck's siren from video clips. I go into a little more depth about these features in this story." \
"The cameras have been upgraded all around, too. The Pixel 8 has a 50-MP main camera that crops into the center to offer a high-quality 2X zoom. There's also a 12-MP ultrawide now with autofocus, allowing it to utilize Google's Macro Focus for close-up shots. " \
"The Pixel 8 Pro has the same main camera but an upgraded 48-MP ultrawide that lets you go even closer to subjects for Macro Focus, and it's overall more effective in low light. You still get a 48-MP telephoto 5X optical zoom camera, and the Pro also exclusively has a front-facing camera with autofocus, allowing for sharper selfies. It's the only one with Pro camera controls in case you want to have more control over your photos, and it will eventually get a new feature called Video Boost, which will send your clips to Google's cloud servers for processing—you'll be sent back footage that is brighter with better stabilization, less noise, and brighter colors. " \
"The 4,485-mAh and 5,050-mAh batteries in the Pixel 8 and Pixel 8 Pro aren't anything to write home about. With average use, you can expect them to last a full day, but anyone using their phone heavily will want to carry a power bank. At least they can recharge slightly faster. "
    
    print("pixel7a: ")
    absa.getMAspects(aspects=["camera", "security", "AI", "design", "battery"], text=pixel7a_review)

    print("pixel8: ")
    absa.getMAspects(aspects=["camera", "security", "AI", "design", "battery"], text=pixel8_review)
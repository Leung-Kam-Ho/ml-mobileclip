import torch
from PIL import Image
import mobileclip
import time 

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/Users/kamholeung/Documents/GitHub/ml-mobileclip/checkpoints/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

im_list = ["nothing_d.png","gate_d.png","gate2_d.png","anson_d.png"]
text = tokenizer(["pipe like structure","human like structure","nothing"])

for im in im_list:
    s = time.time()
    image = preprocess(Image.open(f"docs/{im}").convert('RGB')).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print(f"Label probs {im}:", text_probs, time.time()-s)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm

class SwinTransformerPredictor:

    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=5):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

        self.class_names = [
            "aphids",
            "bacterial_blight",
            "curl_virus",
            "fussarium_wilt",
            "healthy_leaf"
        ]

    def predict(self,image):

        if isinstance(image,str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():

            outputs = self.model(input_tensor)

            probabilities = torch.nn.functional.softmax(outputs[0],dim=0)

            confidence,predicted_class = torch.max(probabilities,0)

        return {
            "prediction": self.class_names[predicted_class.item()],
            "confidence": round(confidence.item()*100,2),
            "probabilities": {
                self.class_names[i]: round(prob.item()*100,2)
                for i,prob in enumerate(probabilities)
            }
        }

    def load_custom_weights(self,weights_path):

        try:
            self.model.load_state_dict(
                torch.load(weights_path,map_location=self.device)
            )
            print("Swin Transformer weights loaded successfully")

        except Exception as e:
            print(f"Error loading weights: {e}")
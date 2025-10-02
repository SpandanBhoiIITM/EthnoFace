import sys
import torch
from torchvision import transforms
import cv2
from model import EthnicityClassifier

def predict(image_path, model_path="models/ethnicity_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = EthnicityClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    ethnicity_map = {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Others"
    }
    return ethnicity_map[pred]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print("Running ethnicity prediction on:", image_path)

    try:
        result = predict(image_path)
        print("✅ Predicted Ethnicity:", result)
    except Exception as e:
        print("❌ Error during prediction:", e)

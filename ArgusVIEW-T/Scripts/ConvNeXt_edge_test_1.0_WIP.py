# ConvNeXt_edge_test_1.0_WIP.py

import os
import time
import torch
import torchvision
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Specify the test model
# test_model = 'convnext_best_model.pth'
# test_model = 'convnext_final_model.pth'
# test_model = 'T1_ConvNeXt_final_model_UF.pth'
# test_model = 'T1_ConvNeXt_best_model_UF.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_10E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_130E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_230E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_330E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_430E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_455E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_530E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_542E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_630E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_6E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_10E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_28E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_50E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_60E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_51E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_119E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_120E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_133E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_140E.pth'
# test_model = 'T2_ConvNeXt_best_model_FZ_16E.pth'
# test_model = 'T2_ConvNeXt_final_model_FZ_20E.pth'
# test_model = 'T2_ConvNeXt_best_model_FZ_13E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_14E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_20E.pth'
# test_model = 'T1_ConvNeXt_best_model_FZ_36E.pth'
# test_model = 'T1_ConvNeXt_final_model_FZ_40E.pth'
# test_model = 'T2_ConvNeXt_final_model_UFZ_250E.pth'
# est_model = 'TBR_T1_ConvNeXt_best_model_UFZ_120E.pth'
# test_model = 'TBR_T1_ConvNeXt_final_model_UFZ_200E.pth'
# test_model = 'TBR_T2_ConvNeXt_best_model_UFZ_161E.pth'
# test_model = 'TBR_T2_ConvNeXt_final_model_UFZ_200E.pth'
# test_model = 'PCR_T1_ConvNeXt_best_model_UFZ_23E.pth'
# test_model = 'PCR_T1_ConvNeXt_final_model_UFZ_200E.pth'
# test_model = 'Hybrid_T1_ConvNeXt_best_model_UFZ_225E.pth'
# test_model = 'Hybrid_T1_ConvNeXt_final_model_UFZ_300E.pth'
test_model = 'Hybrid_T2_ConvNeXt_best_model_UFZ_253E.pth'
# test_model = 'Hybrid_T2_ConvNeXt_final_model_UFZ_300E.pth'


# Define paths
root_dir = 'D:/Hybrid_model_train/'
model_path = os.path.join(root_dir,'dataset_v1.0/T2/saved_models/', test_model)
input_dir = os.path.join(root_dir, 'input/')
result_ok_dir = os.path.join(root_dir, 'results/', 'OK/')
result_ng_dir = os.path.join(root_dir, 'results/', 'NG/')

custom_threshold = 0.3 # Adjust threshold as needed

# Ensure result directories exist
os.makedirs(result_ok_dir, exist_ok=True)
os.makedirs(result_ng_dir, exist_ok=True)

# Load the ConvNeXt model
# pretrained_convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
pretrained_convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
pretrained_convnext.classifier[2] = torch.nn.Linear(in_features=1024, out_features=2)  # previous 2048

# Load the state dict
state_dict = torch.load(model_path, map_location=device, weights_only=True)
print(f'Saved model {test_model} loaded successfully!!!')

pretrained_convnext.load_state_dict(state_dict)
pretrained_convnext.to(device)
pretrained_convnext.eval()

# Define class names
class_names = ['NG', 'OK']

# Define image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_path, threshold=custom_threshold):
    # Predict on the image
    probabilities = predict_image(pretrained_convnext, image_path, class_names, image_transform, device)
    
    # Apply threshold
    pred_class = 'NG' if probabilities[0] >= threshold else 'OK'
    pred_prob = probabilities[0] if pred_class == 'NG' else probabilities[1]
    
    # Move the image to the appropriate result folder
    result_dir = result_ok_dir if pred_class == 'OK' else result_ng_dir
    new_image_path = os.path.join(result_dir, os.path.basename(image_path))
    os.rename(image_path, new_image_path)    
    
    print(f"Processed {os.path.basename(image_path)}: {pred_class} (NG Probability: {probabilities[0]:.3f}, OK Probability: {probabilities[1]:.3f})")

def predict_image(model, image_path, class_names, transform, device):
    img = Image.open(image_path).convert('RGB')
    img_transformed = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_transformed)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    return probabilities.tolist()  # Return probabilities for both classes

def monitor_input_folder():
    while True:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(input_dir, filename)
                process_image(image_path, threshold=custom_threshold)
        
        time.sleep(1)  # Wait for 1 second before checking again

if __name__ == '__main__':
    print("Starting edge test program...")
    print(f"Monitoring input folder: {input_dir}")
    print(f"Results will be saved in: {os.path.dirname(result_ok_dir)}")
    print('Threshold:', custom_threshold)
    print('Waiting.....................................................')
    monitor_input_folder()

import sys
sys.path.append(r"E:\transformer")
from model.vit import vit_base_patch16_224

model_dict = {
    'vit':vit_base_patch16_224,
}

def create_model(model_name, num_classes):   
    return model_dict[model_name](num_classes = num_classes)
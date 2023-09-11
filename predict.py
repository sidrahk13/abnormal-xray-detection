import os
import csv
import sys
import torch
import time
import pandas as pd
from torch.autograd import Variable
from torchvision.datasets.folder import pil_loader
from densenet import densenet169
from torchvision import transforms


def predict(study):

    inputs = Variable(study)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        """need fixing """
    return int(output.data > 0.5)


def load_model():
    model_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)), 'models/model18.pth'
        )
    # print(model_path)
    model = densenet169()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage)['weights']
    )
    return model


def get_transforms():
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        """need fixing """
    return data_transforms


def get_images(study_path):
    study = []
    images = os.listdir(study_path)
    for image in images:
        image_path = os.path.join(study_path, image)
        img = pil_loader(image_path)
        study.append(data_transform(img))
    return study


def get_study_paths_from_csv(input_data_csv_file):
    study_paths = []
    with open(input_data_csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            study_paths.append(row[0].split('image')[0])
    return list(set(study_paths))


def get_predictions(study_paths):
    predictions = pd.DataFrame(columns=['path', 'pred'])
    for i, path in enumerate(study_paths):
        images = get_images(path)
        study = torch.stack(images)
        pred = predict(study)
        predictions.loc[i] = [path, pred]
    return predictions


if __name__ == '__main__':
    import torch._utils
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            """ need fixing here """
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    input_data_csv_file = sys.argv[1]
    output_prediction_csv_path = sys.argv[2]
    """need fixing """
    predictions.to_csv(output_prediction_csv_path, index=False, header=False)
    #accruacy 
    length = 0
    acc =0;
    with open('output.csv','rt')as f:
      data = csv.reader(f)
      for row in data:
        #print(type(row([0])))
        if "positive" in row[0] and row[1]=='1':
            acc+=1
        if "negative" in row[0] and row[1]=='0':
            acc+=1
            
        length+=1

    print(acc/length)    
    
#accrucy 
#0.7648040033361134      

import torch

model = torch.hub.load('pytorch/vision', 'fasterrcnn_resnet50_fpn', pretrained=True)
model.save_model('./model/')
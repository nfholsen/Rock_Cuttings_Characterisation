import subprocess

# 32 batch size for RESNET18 and 8 for RESNET34 to prevent memory leak and the loss to explode

subprocess.call("python train.py -i .\\config\\MAR_RESNET34_CROPPED_256_borehole_train.yaml") # Error in cropped borehole because the images samples only black value due to the shape of the sample ...
subprocess.call("python train.py -i .\\config\\MAR_RESNET34_CROPPED_256_lab_train.yaml")

subprocess.call("python train.py -i .\\config\\MAR_RESNET34_PADDED_256_borehole_train.yaml")
subprocess.call("python train.py -i .\\config\\MAR_RESNET34_PADDED_256_lab_train.yaml")

subprocess.call("python train.py -i .\\config\\MAR_RESNET34_RESIZED_256_borehole_train.yaml")
subprocess.call("python train.py -i .\\config\\MAR_RESNET34_RESIZED_256_lab_train.yaml")

subprocess.call("python train.py -i .\\config\\MAR_RESNET34_RESIZED_256_borehole_train.yaml")
subprocess.call("python train.py -i .\\config\\MAR_RESNET34_RESIZED_256_lab_train.yaml")

subprocess.call("python train.py -i .\\config\\MAR_RESNET18_PADDED_256_borehole_train.yaml")
subprocess.call("python train.py -i .\\config\\MAR_RESNET18_PADDED_256_lab_train.yaml")

subprocess.call("python train.py -i .\\config\\MAR_RESNET18_PADDED_128_borehole_train.yaml")
subprocess.call("python train.py -i .\\config\\MAR_RESNET18_PADDED_128_lab_train.yaml")


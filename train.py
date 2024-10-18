import os
import clip
from clip import model
import tqdm
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# BATCH_SIZE must larger than 1
model_name = "ViT-B/16"
device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
print(f"Training on {device}")
model, preprocess = clip.load(name="ViT-B/16", device=device, jit=False)  # Must set jit=False for training
model = model.cuda()


# checkpoint = torch.load(r"D:\Projects\caches\clip\ViT-B-16.pt")
# # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set
# # context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution  # default is 224
# checkpoint['model_state_dict']["context_length"] = model.context_length  # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size
#
# model.load_state_dict(checkpoint['model_state_dict'])


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt)
        # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title


# By Zhang Tianyi
def get_image_paths(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


# By Zhang Tianyi
def get_image_names(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    images_filenames = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                images_filenames.append(file)
    return images_filenames


# By Zhang Tianyi
def write_images_names(images_filenames, directory):
    with open(directory, 'w') as file:
        for i, filename in enumerate(images_filenames):
            file.write('%s' % filename)
            if i < len(images_filenames) - 1:
                file.write('\n')


# By Zhang Tianyi
def pair_labels(directory):
    full_label = {}
    with open(directory, 'r') as file:
        for key, value in (line.strip().split(' ') for line in file):
            full_label[key] = value
    return full_label


def get_txt_list(image_path, image_dic, full_label):
    text_list = []
    for full_path in image_path:
        value = image_dic[int(full_label[os.path.basename(full_path)])]
        text_list.append(f"This is a {value}.")
    return text_list


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(input_model):
    for p in input_model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# use your own data
EPOCH = 1000
BATCH_SIZE = 96
img_folder = r"D:\Projects\CLIP-train\datasets\part-images"
txt_folder = r"D:\Projects\CLIP-train\datasets\part-images\labels.txt"
img_dic = {0: "pincer plier",  1: "snipe-nose plier", 2: "flat-head screwdriver", 3: "Phillips head screwdriver",
           4: "head of flat-head screwdriver",        5: "head of Phillips head screwdriver",
           6: "hammer",        7: "wrench",           8: "screw",                 9: "nut",
           10: "Allen wrench", 11: "gasket",          12: "cold-pressed nut"}


list_image_path = get_image_paths(img_folder)
list_txt = get_txt_list(list_image_path, img_dic, pair_labels(txt_folder))
# print(list_image_path)
# print("---------------------------")
# print(list_txt)
dataset = image_title_dataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)  # Define your own dataloader

# if device == "cpu":
#     model.float()
# else:
#     clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

# Params used from paper, the lr is smaller, more safe for fine-tuning to new dataset
weight_img = 0.5
weight_text = 1 - weight_img
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

# initialize epoch; total_loss
epoch = 0
total_loss = 0
previous_loss = float("inf")

# add your own code to track the training progress.
for epoch in tqdm(range(EPOCH)):
    print("Epoch {}/{}".format(epoch, EPOCH - 1))
    for batch in train_dataloader:
        optimizer.zero_grad()

        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        previous_loss = total_loss
        total_loss = (weight_img * loss_img(logits_per_image, ground_truth) + weight_text * loss_txt(logits_per_text, ground_truth))
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
    print("Epoch {} Finished. Total Loss: {}".format(epoch, total_loss))

    if total_loss < previous_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total loss': total_loss,
        }, f"D:/Projects/caches/clip/ViT-B-16-Best-New.pt")  # just change to your preferred folder/filename

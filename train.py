import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from datas import ImageDataset
from torch.utils.data import DataLoader
from model import SimpleConvNet
from torch.nn import MSELoss
import torch.optim as opt

#DO NOT FORGET!!! ADD RANDOM SEED

def get_dataset():
    mnist = MNIST(root='./dataset/data', 
                  train=True, 
                  transform=transforms.ToTensor(), 
                  download=True)
    
    #Filter out nine digits
    nine_digits = [img for img, label in mnist if label==9]

    #Use these digits to create custom dataset
    custom_dataset = ImageDataset(nine_digits[:3])

    #Dataloader
    training_data = DataLoader(custom_dataset, batch_size=20, shuffle=True)

    # for batch_features, batch_labels in training_data:
        # print(f"The shape of batch_features is: {batch_features.shape}")

    return training_data

def get_loss(predicted, target):
    loss = MSELoss(reduction='sum')

    # Let's check the grid cells of target and predicted outputs to understand bx, by, bh, bw and conf
    # print(f"The last index of the prediction grid cells is: {predicted_grid_cells[:,0,:,:].shape}")
    # print(f"The last index of the target grid cells is: {batch_labels[:,:,:,0].shape}")

    #Confidence loss using MSE
    #Let's first check if we are using the right cells in target for Confidence loss
    #The right block is the probability of the class - either 0 or 1
    # print(f"The block for the probability of the class in target is: {target[:,:,:,0]}")

    #Similarly let's check if we are using the right cells in predicted for confindence loss
    #The right block is the probability of the class - can be any number between 0 and 1 as it isn't trained yet
    # print(f"The block for the probability of the class in predicted is: {predicted[:,0,:,:]}")

    #Let's compute the confidence loss
    confindence_loss = loss(predicted[:,0,:,:], target[:,:,:,0])

    #Now, for the localization loss, we only use the grid cell that is responsible for object detection
    #To get this grid cell, we Figure out the cell indices of the batch_label that has 1 (as 1 means that grid cell has the highest probability of containing the object)
    responsible_index = torch.nonzero(target[:,:,:,0])

    #Now, that we have the indices of the grid cells that contain the object, let's check the value of these cells in predicted output
    selected_predicted_cells = predicted[responsible_index[:,0], 1:, responsible_index[:,1], responsible_index[:,2]]
    selected_batch_cells = target[responsible_index[:,0], responsible_index[:,1], responsible_index[:,2], 1:]
        
    #Let's recheck if these selected batch_cells are the ones that contain highest probability 1
    # print(f"Rechecking the batch_cells: {selected_batch_cells}")

    localization_output = loss(selected_predicted_cells, selected_batch_cells)

    return confindence_loss + localization_output


def train_model():
    data = get_dataset()

    model = SimpleConvNet()
    optimizer = opt.Adam(model.parameters(), lr=1e-3)

    #We don't need model.train() here, because we don't use dropout or batchnormalization
    # model.train()

    for batch_features, batch_labels in data:

        #So, that gradients are not summed up
        optimizer.zero_grad()

        #Forward pass for model
        predicted_grid_cells = model(batch_features)

        #Compare using loss function
        # print(f"Shape of predicted grid cells is: {predicted_grid_cells.shape}")
        loss = get_loss(predicted_grid_cells, batch_labels)

        #Backward and update gradients
        loss.backward()

        #Optimization using adams
        optimizer.step()

        print(f"The loss is: {loss}")

if __name__ == "__main__":
    torch.manual_seed(42)
    train_model()

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime

torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)


def build_loaders() -> tuple[DataLoader, DataLoader]:
    data_path = 'ml/data-unversioned/'

    label_map = {0: 0, 2: 1, 1: 2}

    cifar10 = datasets.CIFAR10(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        target_transform=lambda label: label_map[label] if label in label_map else label
    )

    cifar10_val = datasets.CIFAR10(
        data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        target_transform=lambda label: label_map[label] if label in label_map else label
    )

    birds_aeroplanes_train = [index for index, sample in enumerate(cifar10) if sample[1] in {0, 1}]
    birds_aeroplanes_val = [index for index, sample in enumerate(cifar10_val) if
                            sample[1] in {0, 1}]

    cifar2 = torch.utils.data.Subset(cifar10, birds_aeroplanes_train)
    cifar2_val = torch.utils.data.Subset(cifar10_val, birds_aeroplanes_val)

    train_loader = DataLoader(cifar2,
                              batch_size=64,
                              shuffle=True)
    val_loader = DataLoader(cifar2_val,
                            batch_size=64,
                            shuffle=False)

    return train_loader, val_loader, cifar2, cifar2_val


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8)  # <1>
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


def training_loop(n_epochs: int,
                  optimizer: torch.optim,
                  model: nn.Module,
                  loss_fn: nn.Module,
                  train_loader: DataLoader,
                  device: torch.device):

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)  # <1>
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))


def validate(model: nn.Module,
             train_loader: DataLoader,
             val_loader: DataLoader,
             device: torch.device,
             cifar2_val
             ):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():


# =============================================================================
#             outputs = model(x_test)
#             predicted = (outputs > 0.5).to(int)
#             total = x_test.shape[0]
#             correct = int((predicted == y_test).sum())
# =============================================================================
# =============================================================================
             for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _ , predicted = torch.max(outputs, dim=1)  # <1>
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
# =============================================================================


        print("Accuracy {}: {:.2f}".format(name, correct / total))
        accdict[name] = correct / total
    return accdict


def main():
    train_loader, val_loader, cifar2, cifar2_val = build_loaders()

    '''
    Showing how a convolution affects the picture
    '''
    len(cifar2)
    img, label = cifar2[3501]

    # Taking a picture from cifar10 dataset
    print(label)
    print(img.size())
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


    # Defining de convolution operation
    conv = nn.Conv2d(in_channels  = 3,
                     out_channels = 16,
                     kernel_size  = 3,
                     stride       = 1,
                     padding      =0
                     )

    # Converting to a 4-dimension Tensor
    img_unsqueezed = img.unsqueeze(0)
    img_output = conv(img_unsqueezed)

    print("Size of tensor previous to the convolution: {0:s}"
          .format(str(img_unsqueezed.size())))
    print("Size of tensor after the convolution: {0:s}"
          .format(str(img_output.size())))


    pool = nn.MaxPool2d(2)

    img_output_pool =   pool(img_unsqueezed)
    print("Size of tensor after the pooling: {0:s}"
          .format(str(img_output_pool.size())))


    '''
    for n in range(0,15):
        plt.imshow(img_output[0,n,:,:].detach())
        plt.show()
    '''

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Training on device {device}.")

    learning_rate = 1e-2

    model = Net().to(device=device)  # <1>
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 80

    training_loop(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device
    )

    torch.save(model, './itsabird.pt')

    all_acc_dict = {"baseline": validate(model, train_loader, val_loader, device, cifar2_val)}

    print(all_acc_dict)


if __name__ == '__main__':
    main()

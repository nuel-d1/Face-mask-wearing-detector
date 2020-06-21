import torch
from torch import nn, optim
import Input
from Model import ConvNet

# Directory to save model checkpoint
model_dir = '/content/detector'

# learning rate
learning_rate = 0.0001

# Loading model
model = ConvNet()

# Confirm that training with gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def training():
    """function for training model"""
    epochs = 10
    steps = 0
    train_loss = []
    print_count = 5

    for epoch in range(epochs):
        for images, labels in Input.train_dataloader:
            # forward pass
            steps += 1
            # moving tensors to gpu
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            train_loss.append(loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            if steps % print_count == 0:
                test_loss, accuracy = validation()

                print('Epoch {}/{} | Training loss: {} | Test Loss: {} | Accuracy: {:.4f} %'
                      .format(epoch + 1, epochs, sum(train_loss) / Input.BATCH_SIZE,
                              sum(test_loss) / len(Input.valid_dataloader),
                              sum(accuracy) / len(Input.valid_dataloader)))

                train_loss = []
    print("\nTraining process is now complete!!")


def validation():
    """function for validation of training results"""

    test_loss = []
    accuracy = []

    for images, labels in Input.valid_dataloader:
        # moving tensors to gpu
        images, labels = images.to(device), labels.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels)
        test_loss.append(loss.item())

        # calculating accuracy
        total = labels.size(0)
        _, prediction = torch.max(output.data, dim=1)
        correct = (prediction == labels).sum().item()
        accuracy.append(correct / total)

    return test_loss, accuracy


def testing():
    "function for testing model"
    with torch.no_grad():
        steps = len(Input.test_dataloader)
        test_loss = []
        accuracy = []

        for batch, (images, labels) in enumerate(Input.test_dataloader):
            # moving tensors to gpu
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss.append(loss.item())

            # calculating accuracy
            total = labels.size(0)
            _, prediction = torch.max(output.data, dim=1)
            correct = (prediction == labels).sum().item()
            accuracy.append(correct / total)

            print("batch {}".format(batch + 1))
            print("\nPrediction accuracy ={:.1f}% "
                  .format((sum(accuracy) / len(Input.test_dataloader) * 100)))


# begin training
training()

# begin testing
testing()

# Save the model state
torch.save(model.state_dict(), 'detector_state.pth')

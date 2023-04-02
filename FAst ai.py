import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the tokenizer
tokenizer = get_tokenizer('basic_english')

# Define the batch size and number of epochs
BATCH_SIZE = 64
NUM_EPOCHS = 5

# Define the training and validation data sets
train_data, val_data = AG_NEWS(root='./data')

# Define the text vocabulary
train_iter = map(lambda x: tokenizer(x[1]), train_data)
vocab = build_vocab_from_iterator(train_iter, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define the collate function
def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor([vocab[token] for token in tokenizer(_text)])
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.long)
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)

# Define the data loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# Define the neural network model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Set the embedding dimension and number of output classes
EMBED_DIM = 32
NUM_CLASSES = len(train_data.get_labels())

# Initialize the model and move it to the device
model = TextClassifier(len(vocab), EMBED_DIM, NUM_CLASSES).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

# Define the TensorBoard summary writer
writer = SummaryWriter()

# Define the training loop function
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for labels, text in iterator:
        optimizer.zero_grad()

        offsets = torch.zeros(len(text) + 1, dtype=torch.long).to(device)
        offsets[1:] = text.sum(1)
        
        predictions = model(text, offsets)
        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Define the evaluation loop function
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

with torch.no_grad():
    for labels, text in iterator:
        offsets = torch.zeros(len(text) + 1, dtype=torch.long).to(device)
        offsets[1:] = text.sum(1)

        predictions = model(text, offsets)
        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

return epoch_loss / len(iterator), epoch_acc / len(iterator)


#Define the accuracy function

def accuracy(predictions, labels):
return (predictions.argmax(1) == labels).float().mean()
Define the main training loop

for epoch in range(NUM_EPOCHS):
train_loss, train_acc = train(model, train_loader, optimizer, criterion)
val_loss, val_acc = evaluate(model, val_loader, criterion)

    def accuracy(predictions, labels):
    return (predictions.argmax(1) == labels).float().mean()
        for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
val_loss, val_acc = evaluate(model, val_loader, criterion)
# Write the loss and accuracy to TensorBoard
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/validation', val_loss, epoch)




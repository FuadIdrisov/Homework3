import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np

### 1.1 Сравнение моделей разной глубины (15 баллов)
# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
num_epochs = 30

# Загрузка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

class ShallowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class DeepNet(nn.Module):
    def __init__(self, num_layers, use_dropout=False, use_batchnorm=False):
        super().__init__()
        layers = []
        input_size = 32*32*3
        
        for i in range(num_layers-1):
            output_size = 512 if i < num_layers-2 else 10
            layers.append(nn.Linear(input_size, output_size))
            
            if i < num_layers-2:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(output_size))
                layers.append(nn.ReLU())
                if use_dropout:
                    layers.append(nn.Dropout(0.5))
            input_size = output_size
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
    
def train_model(model, optimizer, criterion):
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Валидация
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_epoch_loss = test_running_loss / len(test_loader)
        test_epoch_acc = 100. * test_correct / test_total
        test_losses.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_epoch_acc:.2f}%')
    
    training_time = time.time() - start_time
    return train_acc, test_acc, train_losses, test_losses, training_time

depths = [1, 2, 3, 5, 7]
results = {}

for depth in depths:
    print(f"\nTraining model with {depth} layers")
    
    if depth == 1:
        model = ShallowNet().to(device)
    else:
        model = DeepNet(depth).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_acc, test_acc, train_loss, test_loss, time_taken = train_model(model, optimizer, criterion)
    results[depth] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'time': time_taken
    }
    
# Графики точности
plt.figure(figsize=(12, 8))
for depth, res in results.items():
    plt.plot(res['test_acc'], label=f'{depth} layers (Test)')
    plt.plot(res['train_acc'], '--', label=f'{depth} layers (Train)')

plt.title('Accuracy by Model Depth')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# Графики времени обучения
times = [res['time'] for res in results.values()]
plt.bar([str(d) for d in depths], times)
plt.title('Training Time by Model Depth')
plt.xlabel('Number of Layers')
plt.ylabel('Time (seconds)')
plt.show()

# Таблица результатов
print("Model Depth | Train Acc | Test Acc | Time (s)")
for depth, res in results.items():
    print(f"{depth:11} | {max(res['train_acc']):8.2f}% | {max(res['test_acc']):7.2f}% | {res['time']:8.2f}")
    
### 1.2 Анализ переобучения (15 баллов)
# Тестирование регуляризации
depth = 5  # Выбираем модель с 5 слоями (склонную к переобучению)
results_reg = {}

# Базовый вариант (без регуляризации)
model = DeepNet(depth).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_acc, test_acc, _, _, _ = train_model(model, optimizer, nn.CrossEntropyLoss())
results_reg['No Reg'] = {'train': train_acc, 'test': test_acc}

# Вариант с Dropout
model = DeepNet(depth, use_dropout=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_acc, test_acc, _, _, _ = train_model(model, optimizer, nn.CrossEntropyLoss())
results_reg['Dropout'] = {'train': train_acc, 'test': test_acc}

# Вариант с BatchNorm
model = DeepNet(depth, use_batchnorm=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_acc, test_acc, _, _, _ = train_model(model, optimizer, nn.CrossEntropyLoss())
results_reg['BatchNorm'] = {'train': train_acc, 'test': test_acc}

plt.figure(figsize=(12, 8))
for name, res in results_reg.items():
    plt.plot(res['test'], label=f'{name} (Test)')
    plt.plot(res['train'], '--', label=f'{name} (Train)')

plt.title('Regularization Techniques Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
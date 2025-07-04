import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
num_epochs = 30

# Загрузка данных CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Функции для обучения
def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, leave=False)):
        data, target = data.to(device), target.to(device)
        
        if not is_test and optimizer is not None:
            optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(data_loader), correct / total

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%')
        print(f'Test Loss:  {test_loss:.4f}, Acc: {test_acc*100:.2f}%')
        print('-' * 50)
    
    training_time = time.time() - start_time
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'time': training_time
    }

# Модель с настраиваемой шириной слоев
class WidthNet(nn.Module):
    def __init__(self, hidden_sizes, use_dropout=False, use_batchnorm=False):
        super().__init__()
        layers = []
        input_size = 32*32*3
        
        # Скрытые слои
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_size, hidden_size))
            
            if i < len(hidden_sizes) - 1:  # Для всех кроме последнего скрытого
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                if use_dropout:
                    layers.append(nn.Dropout(0.5))
            input_size = hidden_size
        
        # Выходной слой
        layers.append(nn.Linear(input_size, 10))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 2.1 Сравнение моделей разной ширины
def experiment_2_1():
    width_configs = {
        "Узкие": [64, 32, 16],
        "Средние": [256, 128, 64],
        "Широкие": [1024, 512, 256],
        "Очень широкие": [2048, 1024, 512]
    }

    width_results = {}

    print("="*60)
    print("2.1 Сравнение моделей разной ширины")
    print("="*60)

    for name, sizes in width_configs.items():
        print(f"\nОбучение {name.lower()} модели: {sizes}")
        model = WidthNet(sizes).to(device)
        
        # Подсчет параметров
        num_params = model.count_params()
        print(f"Количество параметров: {num_params:,}")
        
        # Обучение модели
        res = train_model(model, train_loader, test_loader, epochs=num_epochs, lr=0.001, device=device)
        
        # Сохранение результатов
        best_train_acc = max(res['train_accs']) * 100
        best_test_acc = max(res['test_accs']) * 100
        width_results[name] = {
            'sizes': sizes,
            'params': num_params,
            'time': res['time'],
            'train_acc': best_train_acc,
            'test_acc': best_test_acc,
            'history': res
        }
        
        print(f"\nРезультаты для {name}:")
        print(f"Точность: train {best_train_acc:.2f}%, test {best_test_acc:.2f}%")
        print(f"Время обучения: {res['time']:.2f} сек")
        print(f"Параметры: {num_params:,}")
        print("="*60)

    # Визуализация результатов
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Точность
    names = list(width_results.keys())
    test_accs = [res['test_acc'] for res in width_results.values()]
    ax[0].bar(names, test_accs, color='skyblue')
    ax[0].set_title('Точность на тестовом наборе')
    ax[0].set_ylabel('Точность (%)')
    ax[0].grid(axis='y')

    # Время обучения
    times = [res['time'] for res in width_results.values()]
    ax[1].bar(names, times, color='lightgreen')
    ax[1].set_title('Время обучения')
    ax[1].set_ylabel('Время (сек)')
    ax[1].grid(axis='y')

    # Количество параметров
    params = [res['params'] for res in width_results.values()]
    ax[2].bar(names, params, color='salmon')
    ax[2].set_title('Количество параметров')
    ax[2].set_ylabel('Параметры')
    ax[2].grid(axis='y')

    plt.tight_layout()
    plt.savefig('width_comparison.png')
    plt.show()

    # Кривые обучения
    plt.figure(figsize=(12, 8))
    for name, res in width_results.items():
        history = res['history']
        plt.plot(np.array(history['test_accs'])*100, label=f'{name} (Test)')
        plt.plot(np.array(history['train_accs'])*100, '--', label=f'{name} (Train)')

    plt.title('Точность моделей разной ширины')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('width_accuracy_curves.png')
    plt.show()
    
    return width_results

# 2.2 Оптимизация архитектуры
def experiment_2_2():
    print("\n" + "="*60)
    print("2.2 Оптимизация архитектуры")
    print("="*60)

    # Варианты размеров слоев
    base_sizes = [128, 256, 512, 1024]
    results_grid = []

    # Перебор всех комбинаций
    for size1 in base_sizes:
        for size2 in base_sizes:
            # Определение схемы
            if size1 > size2:
                scheme = "Сужение"
            elif size1 < size2:
                scheme = "Расширение"
            else:
                scheme = "Постоянная"
            
            print(f"\nОбучение модели: [{size1}, {size2}] ({scheme})")
            model = WidthNet([size1, size2]).to(device)
            
            # Обучение (сокращенное для скорости)
            res = train_model(model, train_loader, test_loader, epochs=15, lr=0.001, device=device)
            
            # Лучшая точность
            best_test_acc = max(res['test_accs']) * 100
            
            # Сохранение результатов
            results_grid.append({
                'size1': size1,
                'size2': size2,
                'scheme': scheme,
                'test_acc': best_test_acc,
                'time': res['time'],
                'params': model.count_params()
            })
            
            print(f"Точность: {best_test_acc:.2f}%, Время: {res['time']:.2f} сек")

    # Создание тепловой карты
    df = pd.DataFrame(results_grid)
    pivot_table = df.pivot_table(values='test_acc', index='size1', columns='size2', aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Точность (%)'})
    plt.title('Точность моделей в зависимости от ширины слоев')
    plt.xlabel('Размер второго слоя')
    plt.ylabel('Размер первого слоя')
    plt.savefig('architecture_heatmap.png')
    plt.show()

    # Анализ по схемам
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='scheme', y='test_acc', data=df)
    plt.title('Сравнение схем изменения ширины')
    plt.xlabel('Схема изменения ширины')
    plt.ylabel('Точность (%)')
    plt.grid(True)
    plt.savefig('width_schemes_comparison.png')
    plt.show()

    # Анализ времени и параметров
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x='params', y='test_acc', hue='scheme', data=df, ax=ax[0], s=100)
    ax[0].set_title('Точность vs Количество параметров')
    ax[0].set_xlabel('Параметры')
    ax[0].set_ylabel('Точность (%)')
    ax[0].grid(True)

    sns.scatterplot(x='time', y='test_acc', hue='scheme', data=df, ax=ax[1], s=100)
    ax[1].set_title('Точность vs Время обучения')
    ax[1].set_xlabel('Время (сек)')
    ax[1].set_ylabel('Точность (%)')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('architecture_efficiency.png')
    plt.show()
    
    return df

# Основная функция
if __name__ == "__main__":
    # Запуск экспериментов
    results_2_1 = experiment_2_1()
    results_2_2 = experiment_2_2()
    
    # Сохранение результатов
    pd.DataFrame.from_dict(results_2_1, orient='index').to_csv('width_results.csv')
    results_2_2.to_csv('architecture_grid_search.csv', index=False)
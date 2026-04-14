import ssl  # SSL証明書の検証を無効にするためのモジュール
ssl._create_default_https_context = ssl._create_unverified_context
import torch    # pytorch本体
import torch.nn as nn   # ニューラルネットワークのモジュール
import torch.optim as optim # 最適化アルゴリズム
import torchvision   # データセットやモデルのライブラリ
import torchvision.transforms as transforms     # データの前処理を行うためのモジュール

# データ読み込み
transform = transforms.Compose([
    transforms.ToTensor(),  # 画像をTensorに変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # ピクセル値を[-1, 1]の範囲に正規化
])

# CIFAR-10データセットのダウンロードと読み込み
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # データセットの作成
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)    # データローダーの作成

# CNNの定義
class SimpleCNN(nn.Module): # nn.Moduleを継承してCNNクラスを定義
    def __init__(self): # コンストラクタ
        super().__init__()  # nn.Moduleのコンストラクタを呼び出す
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # 3チャンネルの入力、16チャンネルの出力、カーネルサイズ3、パディング1の畳み込み層
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)    # 16チャンネルの入力、32チャンネルの出力、カーネルサイズ3、パディング1の畳み込み層
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)    # 32チャンネルの入力、64チャンネルの出力、カーネルサイズ3、パディング1の畳み込み層
        self.pool = nn.MaxPool2d(2, 2)  # カーネルサイズ2、ストライド2の最大プーリング層
        self.fc1 = nn.Linear(64 * 4 * 4, 128)   # 全結合層（入力サイズ64*4*4、出力サイズ128）
        self.fc2 = nn.Linear(128, 10)   # 全結合層（入力サイズ128、出力サイズ10）
        self.relu = nn.ReLU()   # 活性化関数ReLUの定義

    def forward(self, x):   # 順伝播の定義
        x = self.pool(self.relu(self.conv1(x))) # 畳み込み層1、ReLU、プーリングの順で処理
        x = self.pool(self.relu(self.conv2(x))) # 畳み込み層2、ReLU、プーリングの順で処理
        x = self.pool(self.relu(self.conv3(x))) # 畳み込み層3、ReLU、プーリングの順で処理
        x = x.view(-1, 64 * 4 * 4)  # テンソルをフラット化（バッチサイズを維持しつつ、残りの次元を1次元に変換）
        x = self.relu(self.fc1(x))  # 全結合層1とReLUの順で処理
        x = self.fc2(x) # 全結合層2の処理（出力はクラス数10）
        return x    # 出力を返す

# 学習
model = SimpleCNN() # CNNモデルのインスタンス化
criterion = nn.CrossEntropyLoss()   # クロスエントロピー損失関数の定義（分類問題に適した損失関数）
optimizer = optim.Adam(model.parameters(), lr=0.001)    
# Adam最適化アルゴリズムの定義（モデルのパラメータを更新するためのアルゴリズム、学習率0.001）

for epoch in range(3):  # エポック数（データセット全体を何回学習するか）を指定
    running_loss = 0.0  # エポックごとの損失を初期化
    for i, (inputs, labels) in enumerate(trainloader):  # データローダーからバッチごとに入力とラベルを取得
        optimizer.zero_grad()   # 勾配の初期化（前のバッチの勾配をリセット）
        outputs = model(inputs) # モデルに入力を渡して出力を得る
        loss = criterion(outputs, labels)   # 出力とラベルを比較して損失を計算
        loss.backward() # 勾配の計算（損失に対するモデルのパラメータの勾配を計算）
        optimizer.step()    # パラメータの更新（計算された勾配を使用してモデルのパラメータを更新）
        running_loss += loss.item() # 損失を累積
        if i % 200 == 199:  # 200バッチごとに平均損失を表示
            print(f'Epoch {epoch+1}, Step {i+1}, Loss: {running_loss/200:.3f}') # 平均損失を表示
            running_loss = 0.0  # エポックごとの損失をリセット

print("学習完了")   # 学習の完了を表示

# テストデータの読み込み
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  # テストデータセットの作成
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False) # テストデータローダーの作成（シャッフルはFalseに設定して順番を保つ）

# 正解率の計測
correct = 0 # 正解数を初期化
total = 0   # 総数を初期化

model.eval()    # モデルを評価モードに切り替える（ドロップアウトやバッチ正規化などの挙動が変わる）
with torch.no_grad():   # 勾配の計算を無効にする（評価時には勾配は必要ないため、メモリ使用量を削減）
    for inputs, labels in testloader:   #   テストデータローダーからバッチごとに入力とラベルを取得
        outputs = model(inputs) # モデルに入力を渡して出力を得る
        _, predicted = torch.max(outputs, 1)    # 出力の最大値のインデックスを取得（予測されたクラス）
        total += labels.size(0) # ラベルのバッチサイズを総数に加算
        correct += (predicted == labels).sum().item()   # 予測とラベルが一致する数を正解数に加算

print(f"正解率: {100 * correct / total:.2f}%")  # 正解率を計算して表示（正解数を総数で割って100を掛けてパーセンテージ表示）
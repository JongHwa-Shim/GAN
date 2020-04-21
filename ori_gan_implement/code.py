# <코드1> 라이브러리 및 데이터 불러오기

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

#데이터 전처리 방식을 지정한다.
transform = transforms.Compose([
  transforms.ToTensor(), # 데이터를 파이토치의 Tensor 형식으로바꾼다.
  transforms.Normalize(mean=(0.5,), std=(0.5,)) # 픽셀값 0 ~ 1 -> -1 ~ 1
])

#MNIST 데이터셋을 불러온다. 지정한 폴더에 없을 경우 자동으로 다운로드한다.
mnist =datasets.MNIST(root='data', download=True, transform=transform)

#데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.
dataloader =DataLoader(mnist, batch_size=60, shuffle=True)

import os
import imageio

if torch.cuda.is_available():
    use_gpu = True
leave_log = True
if leave_log:
    result_dir = 'GAN_generated_images'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)


# <코드2> GAN의 생성자(Generator)

# 생성자는 랜덤 벡터 z를 입력으로 받아 가짜 이미지를 출력한다.
class Generator(nn.Module):

  # 네트워크 구조
    def __init__(self):
      super(Generator, self).__init__()
      self.main = nn.Sequential(
        nn.Linear(in_features=100, out_features=256),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=256, out_features=512),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=512, out_features=1024),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=1024, out_features=28*28),
        nn.Tanh())
    
  # (batch_size x 100) 크기의 랜덤 벡터를 받아 
  # 이미지를 (batch_size x 1 x 28 x 28) 크기로 출력한다.
    def forward(self, inputs):
      return self.main(inputs).view(-1, 1, 28, 28) #(batch_size, channel, 가로, 세로)

# <코드3> GAN의 구분자(Discriminator)

# 구분자는 이미지를 입력으로 받아 이미지가 진짜인지 가짜인지 출력한다.
class Discriminator(nn.Module):
    
# 네트워크 구조
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      nn.Linear(in_features=28*28, out_features=1024),
      nn.LeakyReLU(0.2),
      nn.Dropout(inplace=True),
      nn.Linear(in_features=1024, out_features=512),
      nn.LeakyReLU(0.2),
      nn.Dropout(inplace=True),
      nn.Linear(in_features=512, out_features=256),
      nn.LeakyReLU(0.2),
      nn.Dropout(inplace=True),
      nn.Linear(in_features=256, out_features=1),
      nn.Sigmoid())
    
  # (batch_size x 1 x 28 x 28) 크기의 이미지를 받아
  # 이미지가 진짜일 확률을 0~1 사이로 출력한다.
  def forward(self, inputs):
    inputs = inputs.view(-1, 28*28)
    return self.main(inputs)

# <코드4> 생성자와 구분자 객체 만들기

G = Generator()
G = G.cuda()
D = Discriminator()
D = D.cuda()

# <코드5> 손실 함수와 최적화 기법 지정하기

# Binary Cross Entropy loss
criterion = nn.BCELoss()

# 생성자의 매개 변수를 최적화하는 Adam optimizer
G_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 구분자의 매개 변수를 최적화하는 Adam optimizer
D_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# <코드6> 모델 학습을 위한 반복문

# 데이터셋을 100번 돌며 학습한다.
number = range(200)
z_fixed = Variable(torch.randn((1, 100)))
z_fixed = z_fixed.cuda()
for epoch in number:

  # 한번에 batch_size만큼 데이터를 가져온다.
    for real_data, _ in dataloader:
      batch_size = real_data.size(0)
        
      # 데이터를 파이토치의 변수로 변환한다.
      real_data = Variable(real_data)
      real_data = real_data.cuda()
      # ...(중략)

# <코드7> 구분자 학습시키기

      # 이미지가 진짜일 때 정답 값은 1이고 가짜일 때는 0이다.
      # 정답지에 해당하는 변수를 만든다.
      target_real = Variable(torch.ones(batch_size, 1)).cuda()
      target_fake = Variable(torch.zeros(batch_size, 1)).cuda()

      # 진짜 이미지를 구분자에 넣는다.
      D_result_from_real = D(real_data)

      # 구분자의 출력값이 정답지인 1에서 멀수록 loss가 높아진다.
      D_loss_real = criterion(D_result_from_real, target_real)

      # 생성자에 입력으로 줄 랜덤 벡터 z를 만든다.
      z = Variable(torch.randn((batch_size, 100)))
      z = z.cuda()
      # 생성자로 가짜 이미지를 생성한다.
      fake_data = G(z)

      # 생성자가 만든 가짜 이미지를 구분자에 넣는다.
      D_result_from_fake = D(fake_data)

      # 구분자의 출력값이 정답지인 0에서 멀수록 loss가 높아진다.
      D_loss_fake = criterion(D_result_from_fake, target_fake)

      # 구분자의 loss는 두 문제에서 계산된 loss의 합이다.
      D_loss = D_loss_real + D_loss_fake

      # 구분자의 매개 변수의 미분값을 0으로 초기화한다.
      D.zero_grad()

      # 역전파를 통해 매개 변수의 loss에 대한 미분값을 계산한다.
      D_loss.backward()

      # 최적화 기법을 이용해 구분자의 매개 변수를 업데이트한다.
      D_optimizer.step()

  # <코드8> 생성자 학습시키기

      # 생성자에 입력으로 줄 랜덤 벡터 z를 만든다.
      z = Variable(torch.randn((batch_size, 100)))
      z = z.cuda()

      # 생성자로 가짜 이미지를 생성한다.
      fake_data = G(z)

      # 생성자가 만든 가짜 이미지를 구분자에 넣는다.
      D_result_from_fake = D(fake_data)

      # 생성자의 입장에서 구분자의 출력값이 1에서 멀수록 loss가 높아진다.
      G_loss = criterion(D_result_from_fake, target_real)

      # 생성자의 매개 변수의 미분값을 0으로 초기화한다.
      G.zero_grad()

      # 역전파를 통해 매개 변수의 loss에 대한 미분값을 계산한다.
      G_loss.backward()

      # 최적화 기법을 이용해 생성자의 매개 변수를 업데이트한다.
      G_optimizer.step()

    fake_data_fixed = G(z_fixed)
    arr = fake_data_fixed[0,0,0:28,0:28]
    arr = arr.cpu().detach().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    plt.imshow(arr)
    name = 'epoch_' + str(epoch)
    plt.savefig(name)

    print("D loss:", D_loss, "  G loss:", G_loss,"\n")
    print(epoch)
"""
arr = fake_data[0,0,0:28,0:28]
arr = arr.cpu().detach().numpy()
plt.imshow(arr)
plt.show()
"""
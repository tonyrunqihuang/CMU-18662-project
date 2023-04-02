import torch

if __name__ == '__main__':
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(DEVICE)
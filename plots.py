import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m','--mode',type=str,required=True,help='either "test" or "train"')

args = parser.parse_args()

a = np.load(f'stock_rewards/{args.mode}.npy')

print(f"average rewards : {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

if args.mode == 'train':
  # show the training progress
  plt.plot(a)
else:
  # test - show a histogram of rewards
  plt.hist(a, bins=20)

plt.title(args.mode)
plt.show()
import pickle
import numpy
from collections import defaultdict

groundtruth_path = 'utils/groundtruth_200.p'
with open(groundtruth_path, 'rb') as file:
        groundtruth = pickle.load(file)

print(len(groundtruth.items()))

counter = 0
groundtruth_200 = defaultdict(list)
for item in groundtruth.items():
	key = item[0]
	value = item[1]
	groundtruth_200[key] = value
	counter += 1
	if counter == 199:
		break
	print(key)
print("Counter: ",counter)

#print(len(groundtruth_200))

with open('utils/groundtruth_200.p', 'wb') as f:
	pickle.dump(groundtruth_200, f)

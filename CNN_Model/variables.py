cat_folder = "Cat"
dog_folder = "Dog"
cutoff = 0.8
input_size = 50

in_channels = 1
ofm1 = 6
ofm2 = 12
dense1 = 512
dense2 = 64
output = 1
kernal_size = 5
kernal_pool = 2
stride = 2

batch_size = 100
num_epochs = 10
learning_rate = 0.001

encoder = {"cat": 0, "dog": 1}
train_data = 'TrainData.npy'
test_data = 'TestData.npy'

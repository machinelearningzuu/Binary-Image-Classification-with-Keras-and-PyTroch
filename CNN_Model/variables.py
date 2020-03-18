cat_folder = "Cat"
dog_folder = "Dog"
cutoff = 0.8
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
current_dir = "E:\My Projects\Cat-vs-Dog-Image-Classifier-Differnet-Models-with-Pytroch\PetImages"


# Variables specific for cnn.py script
# in_channels = 1
# crop = 50
# input_size = in_channels * crop * crop
# encoder = {"cat": 0, "dog": 1}
# train_data = 'TrainData.npy'
# test_data = 'TestData.npy'

# Variables specific for cnn_with_dataloader.py script
crop = 224
in_channels = 3
workers = 4
input_size = in_channels*crop*crop # for CatvsDogs_with_dataloader.py

# Common Variables
cat_folder = "Cat"
dog_folder = "Dog"
cutoff = 0.8
batch_size = 128
dense1 = 512
dense2 = 64
dense3 = 64
output = 1
keep_prob = 0.3
learning_rate = 0.0001
num_epochs = 8
verbose = 100
current_dir = "E:\My Projects\Cat-vs-Dog-Image-Classifier-Differnet-Models-with-Pytroch\PetImages"

# Variables specific for CatvsDogs.py script
# crop = 50
# input_size = crop * crop
# encoder = {"cat": 0, "dog": 1}
# train_data = 'TrainData.npy'
# test_data = 'TestData.npy'

# Variables specific for CatvsDogs_with_dataloader.py script
crop = 224
input_channels = 3
workers = 4
input_size = input_channels*crop*crop # for CatvsDogs_with_dataloader.py

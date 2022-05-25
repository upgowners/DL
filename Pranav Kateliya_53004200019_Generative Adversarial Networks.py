import torch #import the PyTorch library with torch.
from torch import nn # import nn just to be able to set up the neural networks in a less verbose way.

import math # import math to obtain the value of the pi constant.
import matplotlib.pyplot as plt #import the Matplotlib plotting tools as plt as usual.

#The number 111 represents the random seed used to initialize the random number generator, which is used to initialize the neural network’s weights. 
#Despite the random nature of the experiment, it must provide the same results as long as the same seed is used.
torch.manual_seed(111)

#The training data is composed of pairs (x₁, x₂) so that x₂ consists of the value of the sine of x₁ for x₁ in the interval from 0 to 2π.
train_data_length = 1024 #training set is composed with 1024 pairs (x₁, x₂).

#A tensor is a multidimensional array similar to a NumPy array.
train_data = torch.zeros((train_data_length, 2)) # initializing train_data, a tensor with dimensions of 1024 rows and 2 columns, all containing zeros.

train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length) #Using the first column of train_data to store random values in the interval from 0 to 2π.
train_data[:, 1] = torch.sin(train_data[:, 0]) #Calculating the second column of the tensor as the sine of the first column.
train_labels = torch.zeros(train_data_length) #Create train_labels, a tensor filled with zeros.

#Create train_set as a list of tuples, with each row of train_data and train_labels represented in each tuple as expected by PyTorch’s data loader.
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

#Plotting training data at each point (x₁, x₂)
plt.plot(train_data[:, 0], train_data[:, 1], ".")

#Creating a PyTorch data loader with train_set
#Creating a data loader called train_loader, which will shuffle the data from train_set and return batches of 32 samples that you’ll use to train the neural networks.
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

#In PyTorch, the neural network models are represented by classes that inherit from nn.Module 
#The discriminator is a model with a two-dimensional input and a one-dimensional output. 
#It’ll receive a sample from the real data or from the generator and will provide the probability that the sample belongs to the real training data. 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__() #Calling super().__init__() to run .__init__() from nn.Module.
        #The discriminator you’re using is an MLP neural network defined in a sequential way using nn.Sequential().
        self.model = nn.Sequential( 
            #The input is two-dimensional, and the first hidden layer is composed of 256 neurons with ReLU activation.
            nn.Linear(2, 256),
            nn.ReLU(),
            # After the first, second, and third hidden layers, you use dropout to avoid overfitting.
            nn.Dropout(0.3),
            #The second and third hidden layers are composed of 128 and 64 neurons, respectively, with ReLU activation.
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            #The output is composed of a single neuron with sigmoidal activation to represent a probability.
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    #Using .forward() to describe how the output of the model is calculated. 
    #x represents the input of the model, which is a two-dimensional tensor.
    def forward(self, x):
        output = self.model(x)
        return output

#discriminator represents an instance of the neural network you’ve defined and is ready to be trained.
discriminator = Discriminator()

#Implementing the Generator, The generator is the model that takes samples from a latent space as its input and generates data resembling the data in the training set. 
#In this case, it’s a model with a two-dimensional input, which will receive random points (z₁, z₂), and a two-dimensional output that must provide (x̃₁, x̃₂) points resembling those from the training data.
#The generator is composed of two hidden layers with 16 and 32 neurons, both with ReLU activation, and a linear activation layer with 2 neurons in the output. 
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

lr = 0.001 #Setting the learning rate (lr), which is use to adapt the network weights.

num_epochs = 300 #Setting the number of epochs (num_epochs), which defines how many repetitions of training using the whole training set will be performed.

#The binary cross-entropy function is a suitable loss function for training the discriminator because it considers a binary classification task. 
#It is also suitable for training the generator since it feeds its output to the discriminator, which provides a binary observable output.
loss_function = nn.BCELoss() #Assigning the variable loss_function to the binary cross-entropy function BCELoss(), which is the loss function that you’ll use to train the models.

#PyTorch implements various weight update rules for model training in torch.optim.
#Using the Adam algorithm to train the discriminator and generator models.
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        #Using torch.ones() to create labels with the value 1 for the real samples, and then assigning it to the labels of real_samples_labels.
        real_samples_labels = torch.ones((batch_size, 1))
        #Creating the generated samples by storing random data in latent_space_samples, which is then feed to the generator to obtain generated_samples.
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        #Using torch.zeros() to assign the value 0 to the labels for the generated samples, and then you store the labels in generated_samples_labels.
        generated_samples_labels = torch.zeros((batch_size, 1))
        #Concatenating the real and generated samples and labels and store them in all_samples and all_samples_labels
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        #In PyTorch, it is necessary to clear the gradients at each training step to avoid accumulating them using .zero_grad().
        discriminator.zero_grad()
        #Calculating the output of the discriminator using the training data in all_samples.
        output_discriminator = discriminator(all_samples)
        #Calculating the loss function using the output from the model in output_discriminator and the labels in all_samples_labels.
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        #Calculating the gradients to update the weights with loss_discriminator.backward().
        loss_discriminator.backward()
        #Updating the discriminator weights by calling optimizer_discriminator.step().
        optimizer_discriminator.step()

        # Data for training the generator
        #Storing random data in latent_space_samples, with a number of lines equal to batch_size.
        latent_space_samples = torch.randn((batch_size, 2))

        # Training the generator
        generator.zero_grad()
        #Feeding the generator with latent_space_samples and store its output in generated_samples.
        generated_samples = generator(latent_space_samples)
        #Feeding the generator’s output into the discriminator and store its output in output_discriminator_generated, which you’ll use as the output of the whole model.
        output_discriminator_generated = discriminator(generated_samples)
        #Calculating the loss function using the output of the classification system stored in output_discriminator_generated and the labels in real_samples_labels, which are all equal to 1.
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        #Calculating the gradients and update the generator weights.
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        #Displaying the values of the discriminator and generator loss functions at the end of each ten epochs.
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)

#Using .detach() to return a tensor from the PyTorch computational graph.
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
import torch
import torch.nn
import torch.optim

class Torch:
    def __init__(self):
        self.torch = torch
    
    def is_cuda():
        return torch.cuda.is_available()


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_, output_, hidden_layers, hidden_layers_input_output):
        super(NeuralNetwork, self).__init__()
        self.inout = [input_, output_]
        self.layers_hidden_count = hidden_layers
        self.hidden_layers_input_output = hidden_layers_input_output
        self.layers_hidden = []
        self.activations = []

        self.activation_function = {"Sigmoid": torch.nn.Sigmoid}["Sigmoid"]
        self.SimpleNN = False #Константа, не доработано

        if not self.SimpleNN:
            self.input_layer = torch.nn.Linear(input_, hidden_layers_input_output)
            self.input_activation = torch.nn.Sigmoid()
            self.output_layer = torch.nn.Linear(hidden_layers_input_output, output_)
        else:
            self.input_layer = torch.nn.Linear(input_, output_)
            self.input_activation = torch.nn.Sigmoid()

        for _ in range(hidden_layers):
            self.layers_hidden.append(torch.nn.Linear(hidden_layers_input_output, hidden_layers_input_output))
            self.activations.append(torch.nn.Sigmoid())


    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)

        for indx_layer_activation in range(self.layers_hidden_count):
            x = self.layers_hidden[indx_layer_activation](x)
            x = self.activations[indx_layer_activation](x)

        if not self.SimpleNN:
            x = self.output_layer(x)

        return x


class Training:
    def __init__(self, model, optimizer: str, learning_rate = 0.001, criterion = "MSELoss"):
        self.model = model

        self.optimizer_name = optimizer
        self.criterion_name = criterion

        self.optimizer = {
            'Adam': torch.optim.Adam(model.parameters(), lr = learning_rate),
            'SGD': torch.optim.SGD(model.parameters(), lr = learning_rate) 
        }[optimizer]

        self.criterion = {
            "MSELoss": torch.nn.MSELoss()
        }[criterion]

        self.base_file = None
        self.mddloss = 0


    def get_base(self):
        all_strings = self.base_file.split("\n")
        all_info = len(all_strings)
        input_val = list([list(map(int, all_strings[x].split(","))) for x in range(all_info) if x % 2 == 0])
        target_val = list([list(map(int, all_strings[x].split(","))) for x in range(all_info) if x % 2 == 1])
        return input_val, target_val


    def set_train_val(self, input_val, target_val):
        self.input_val = torch.Tensor(input_val)
        self.target_val = torch.Tensor(target_val)
        #print(self.input_val.shape, self.target_val.shape)


    def train(self, epochs = 100):
        all_loss = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            output = self.model(self.input_val)
            loss = self.criterion(output, self.target_val)
            all_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()

        return all_loss


    def return_model(self):
        return self.model
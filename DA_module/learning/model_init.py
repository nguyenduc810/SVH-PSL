import torch
from DA_module.learning.utils import prepare_dom_data, compute_class_weight, load_batched_dom_data, train_nn
from DA_module.learning.model import NeuralNet
import torch.nn as nn



def init_dom_nn_classifier(x, y, rel_map, dom, problem,batch_size=32,
                           activation='relu', lr=0.001, weight_decay=0.00001, device =None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    input_size = 2 * problem.n_var
    hidden_size = 200
    num_hidden_layers = 3
    epochs = 20
    data = prepare_dom_data(x, y, rel_map, dom, data_kind='tensor', device=device)
    weight = compute_class_weight(data[:, -1])
    # print(data[:, -1])
    if weight is None:
        return None

    net = NeuralNet(input_size, hidden_size, num_hidden_layers,
                    activation=activation).to(device)

    weight = torch.tensor(weight, device=device).float()
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_nn(data, load_batched_dom_data, net, criterion, optimizer, batch_size, epochs)

    return net


# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

#   description: |
#     - Load in the Boston housing dataset from sklearn
X, y = datasets.load_boston(return_X_y=True)
X.shape
# Normalize the data
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
X = torch.tensor(X).float()
y = torch.tensor(y).float()
y = y.reshape(-1, 1)

# %%
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(13, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 1),
        )

    def forward(self, x):
        """torch.nn.Module auto call forward"""
        return self.layers(x)


# %%
mymodel = NeuralNetwork()
mymodel.parameters()  # aggragate parameters from every layer
print(len(list(mymodel.parameters())[0]))

pred = mymodel.forward(X)

# %%
def train(model, X, y, epochs=300):
    losses = []
    scores = []
    # create optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        # for batch in batches:
        y_hat = model(X)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()  # reset the grad of each param
        r2_score = metrics.r2_score(y_hat.detach().numpy(), y.detach().numpy())
        print("loss:", loss)
        print("r2_score:", r2_score)
        losses.append(loss.detach().numpy())
        scores.append(r2_score)

    # fig, axs = plt.subplots(2)
    # axs[0].plot(losses)
    # axs[1].plot(scores)
    # fig.suptitle('Vertically stacked subplots')
    # plt.plot(losses)
    # plt.plot(scores)
    # # plt.xlim(0, 300)
    # # plt.ylim(0, 20)
    # # plt.subplot()
    # plt.show()


train(mymodel, X, y)
# %%


# def create_dummy_file():
#     features = "rooms, zipcode, median_price, school_rating, transport"
#     with open("features.txt", "w") as f:
#         f.write(features)


# create_dummy_file()

# Create experiment (artifact_location=./ml_runs by default)
# mlflow.set_experiment("My Dummy Project")

# By default experiment we've set will be used
with mlflow.start_run():
    # mlflow.log_artifact("features.txt")
    # knn_parameters = {
    #     "n_neighbors": list(range(1, 30)),
    #     "weights": ["uniform", "distance"],
    # }

    # for i in range(10):
    #     mlflow.log_params(knn_parameters)

    mlflow.sklearn.log_model(
        sk_model=mymodel,
        artifact_path="sklearn-model",
        registered_model_name="sklearn-regression",
    )
    # for i in range(10):
    #     mlflow.log_metric("Iteration", i, step=i)

# mlflow server --backend-store-uri sqlite:///\mlflow.db --default-artifact-root ./artifact --host 127.0.0.1
# split bash
# export MLFLOW_TRACKING_URI=http://localhost:5000
# mlflow run .
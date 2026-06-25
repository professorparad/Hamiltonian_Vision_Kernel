import matplotlib.pyplot as plt


def plot_training_curves(total_loss, reconstruction_loss, energy_loss):

    steps = range(len(total_loss))

    plt.figure(figsize=(8, 5))

    plt.plot(steps, total_loss, label="Total Loss")

    plt.plot(steps, reconstruction_loss, label="Reconstruction Loss")

    plt.plot(steps, energy_loss, label="Energy Loss")

    plt.xlabel("Training Step")

    plt.ylabel("Loss")

    plt.title("HVK Training Curves")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

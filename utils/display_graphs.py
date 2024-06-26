import matplotlib.pyplot as plt
import sys

def display_graphs(validation_loss, validation_accuracy, training_loss, training_accuracy):
    plt.figure(1)
    plt.plot(validation_loss, label="Validation Loss")
    plt.plot(training_loss, label="Training Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block=False)
    plt.waitforbuttonpress()

    plt.figure(2)
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.plot(training_accuracy, label="Training Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.waitforbuttonpress()
    plt.close()


def main():
    validation_loss = list(map(float, sys.argv[1].split()))
    validation_accuracy = list(map(float, sys.argv[2].split()))
    training_loss = list(map(float, sys.argv[3].split()))
    training_accuracy = list(map(float, sys.argv[4].split()))

    display_graphs(validation_loss, validation_accuracy, training_loss, training_accuracy)


if __name__ == "__main__":
    main()
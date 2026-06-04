import time
import tracemalloc
from meditation.params import *
import matplotlib.pyplot as plt

def simple_time_and_memory_tracker(method):

    # ### Log Level
    # 0: Nothing
    # 1: Print Time and Memory usage of functions
    LOG_LEVEL = 1

    def method_with_trackers(*args, **kw):
        ts = time.time()
        tracemalloc.start()
        result = method(*args, **kw)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        te = time.time()
        duration = te - ts
        if LOG_LEVEL > 0:
            output = f"{method.__qualname__} executed in {round(duration, 2)} seconds, using up to {round(peak / 1024**2,2)}MB of RAM"
            print(output)
        return result

    return method_with_trackers


def plot_loss_accuracy(history,ymax_loss_, ymax_accuracy):
    # Setting figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))

    # Create the plots
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])

    # Set titles and labels
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    ax2.set_title('accuracy')
    ax2.set_ylabel('val_accuracy')
    ax2.set_xlabel('Epoch')

    # Set limits for y-axes
    ax1.set_ylim(ymin=0, ymax=ymax_loss_)
    ax2.set_ylim(ymin=0, ymax=ymax_accuracy)

    # Generate legends
    ax1.legend(['Train', 'Validation'], loc='best')
    ax2.legend(['Train', 'Validation'], loc='best')

    # Show grids
    ax1.grid(axis="x", linewidth=0.5)
    ax1.grid(axis="y", linewidth=0.5)

    ax2.grid(axis="x", linewidth=0.5)
    ax2.grid(axis="y", linewidth=0.5)

    plt.show()

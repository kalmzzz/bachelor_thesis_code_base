import time
from train_methods import *

# ---------------- Parameters -----------------------
EPOCHS = 50
LR = 0.1
EPS = 0.0314
ITER = 7
ALPHA = 0.00784
BATCH_SIZE = 128

# ---------------------------------------------------

if __name__ == "__main__":
    start = time.perf_counter()

    train(epochs=EPOCHS,
          learning_rate=LR,
          output_name="basic_training_single_cat_to_dog",
          data_suffix="single_cat_to_dog",
          batch_size=BATCH_SIZE,
          data_augmentation=False
          )

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

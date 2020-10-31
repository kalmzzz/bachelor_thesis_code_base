import time
from train_methods import *

# ---------------- Parameters -----------------------
EPOCHS = 100
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
          complex=False,
          output_name="basic_training_single_cat_to_auto",
          data_suffix="single_cat_to_auto",
          batch_size=BATCH_SIZE,
          data_augmentation=True
          )

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

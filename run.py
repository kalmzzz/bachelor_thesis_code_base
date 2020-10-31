import time
from train_methods import *
from evaluation_methods import *
from dataset_generation_methods import *

# ---------------- Parameters -----------------------
EPS = 0.5
ITER = 10

EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128

# Target Class wird als new class erkannt während new class normal erkannt wird
TARGET_CLASS = 7
NEW_CLASS = 4
# ---------------------------------------------------

if __name__ == "__main__":
    start = time.perf_counter()

# -------------------- Dataset Generation -----------------------

    #generate_pertubed_dataset_main(eps=EPS, iter=ITERS, target_class=TARGET_CLASS, new_class=NEW_CLASS, dataset_name="auto_to_deer", inf=False, pertube_count=1.0)

    generate_single_image_pertubed_dataset(model_path="basic_training", output_name="single_horse_to_deer", target_class=TARGET_CLASS, new_class=NEW_CLASS, EPS=EPS, ITERS=ITERS, pertube_count=1.0)

# ------------------- Training --------------------------------

    train(epochs=EPOCHS,
          learning_rate=LR,
          complex=False,
          output_name="basic_training_single_horse_to_deer",
          data_suffix="single_horse_to_deer",
          batch_size=BATCH_SIZE,
          data_augmentation=True
          )

# ------------------- Evaluation -----------------------------

    evaluate_single_class(model_path="basic_training_single_horse_to_deer", target_class=TARGET_CLASS, new_class=NEW_CLASS)

    #single_model_evaluation(model_path="basic_training")

    #analyze_layers(EPS, ITER, target_class=TARGET_CLASS, new_class=NEW_CLASS)

    #analyze_general_layer_activation(target_class=TARGET_CLASS)


    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

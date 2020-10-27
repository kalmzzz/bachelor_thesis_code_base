import time
from evaluation_methods import *

# ---------------- Parameters -----------------------
EPS = 0.5
ITER = 10
# ---------------------------------------------------

if __name__ == "__main__":
    start = time.perf_counter()

    #single_model_evaluation(model_path="basic_training_single_cat_to_dog")

    analyze_layers(EPS, ITER, target_class=3, new_class=5)

    analyze_general_layer_activation(target_class=3)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.)
    print(f"Computation time: {duration:0.4f} Minutes")

import time
from dataset_generation_methods import *

# ---------------- Parameters -----------------------
EPS = 2.0
ITER = 24
# ---------------------------------------------------

if __name__ == "__main__":
    start = time.perf_counter()

    #generate_pertubed_dataset_main(eps=EPS, iter=ITER, target_class=3, new_class=5, inf=False, pertube_count=1.0)

    generate_single_image_pertubed_dataset(model_path="basic_training", target_class=3, new_class=5, pertube_count=1.0)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

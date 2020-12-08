import time
import os
from train_methods import *
from evaluation_methods import *
from dataset_generation_methods import *
import torch
import torch.backends.cudnn as cudnn
import numpy as np
AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
BCE, WASSERSTEIN, KLDIV = 0, 1, 2

# torch.manual_seed(42)
# np.random.seed(42)
# torch.cuda.manual_seed(42)
cudnn.benchmark = True
# cudnn.deterministic=True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------- Parameters -----------------------

EPS = 0.5 #epsilon
ITERS = 100 #wiederholungen von pgd

EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128

PERT_COUNT = 0.5 #gibt an wieviel Prozent der Zielklasse "perturbed" sein sollen

PERT_COUNT_GRADS = 0.3 #gibt an wieviel Prozent der Zielklasse "perturbed" sein sollen | für gradienten methode
GRADIENT_THRESHOLD = 0.6 #threshold ab welchem die gradienten benutzt werden

# Target Class wird als new class erkannt während new class normal erkannt wird
TARGET_CLASS = DEER
NEW_CLASS = AUTO

LOSS_FN = KLDIV

DATASET_NAME = "single_deer_to_auto_kldiv_no_softmax"
CUSTOM_BEST_IMAGE_ID = None #wenn nicht das beste bild genommen werden soll, kann man hier eine wunsch id einsetzen

# ---------------------------------------------------

if __name__ == "__main__":
    start = time.perf_counter()

# -------------------- Dataset Generation -----------------------

    #generate_pertubed_dataset_main(eps=EPS, iter=ITERS, target_class=TARGET_CLASS, new_class=NEW_CLASS, dataset_name=DATASET_NAME, inf=False, pertube_count=PERT_COUNT)
    best_image_id = CUSTOM_BEST_IMAGE_ID
    best_image_id = generate_single_image_pertubed_dataset(model_name="basic_training", output_name=DATASET_NAME, target_class=TARGET_CLASS, new_class=NEW_CLASS, EPS=EPS, ITERS=ITERS, pertube_count=PERT_COUNT, loss_fn=LOSS_FN, custom_id=CUSTOM_BEST_IMAGE_ID)
    #best_image_id = generate_single_image_pertubed_dataset_gradients(output_name=DATASET_NAME, target_class=TARGET_CLASS, new_class=NEW_CLASS, pertube_count=PERT_COUNT_GRADS, gradient_threshold=GRADIENT_THRESHOLD)

# ------------------- Training --------------------------------

    train(epochs=EPOCHS,
          learning_rate=LR,
          complex=False,
          output_name="basic_training_"+str(DATASET_NAME),
          #output_name="basic_training_non_robust_no_softmax",
          #data_suffix="L2_"+str(DATASET_NAME)+"_"+str(PERT_COUNT)+"pert_"+str(ITERS)+"iters_"+str(EPS)+"eps",
          data_suffix=DATASET_NAME,
          batch_size=BATCH_SIZE,
          data_augmentation=True
          )

# ------------------- Evaluation -----------------------------
    result_path = 'results/'+str(DATASET_NAME)+'_results'

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    analyze_layers(EPS, ITERS, target_class=TARGET_CLASS, new_class=NEW_CLASS, save_path=result_path, model_name="basic_training_"+str(DATASET_NAME), target_id=best_image_id, pert_count=PERT_COUNT, grad_thresh=GRADIENT_THRESHOLD, loss_fn=LOSS_FN)

    evaluate_single_class(model_name="basic_training_"+str(DATASET_NAME), save_path=result_path, target_class=TARGET_CLASS, new_class=NEW_CLASS)

    single_model_evaluation(model_name="basic_training_"+str(DATASET_NAME), save_path=result_path)

    #analyze_general_layer_activation(target_class=TARGET_CLASS)


    print("finished: [ " + str(DATASET_NAME) + " ]")
    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

import time
import os
from train_methods import *
from evaluation_methods import *
from dataset_generation_methods import *
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sys

cudnn.benchmark = True


# ---------------- Parameters -----------------------
AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
loss_dict = {0:"BCE_WithLogits", 1:"Wasserstein", 2:"KLDiv"}
BCE, WASSERSTEIN, KLDIV = 0, 1, 2
ITERS = 100 #wiederholungen von pgd
EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128

# ---------------------------------------------------

if __name__ == "__main__":
    start = time.perf_counter()

    if sys.argv is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if sys.argv[1] == "0":
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if sys.argv[1] == "1":
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    print("Start Calculating on Device: " + str(device))

    epsilon = [2.0, 1.0, 0.75, 0.5, 0.25, 0.1]
    perturbation_counts = [0.5]#[1.0, 0.75, 0.5, 0.25] #gibt an wieviel Prozent der Zielklasse "perturbed" sein sollen
    loss_functions = [BCE, WASSERSTEIN]
    layer_cuts = [2, 1]
    TARGET_CLASS = DEER # Target Class wird als new class erkannt während new class normal erkannt wird
    NEW_CLASS = HORSE
    CUSTOM_BEST_IMAGE_ID = 9035 #wenn nicht das beste bild genommen werden soll, kann man hier eine wunsch id einsetzen
    #LAYER = 2 #wieviele Dense Layer abgeschnitten werden sollen

    SAVE_PATH = "single_"+str(class_dict[TARGET_CLASS]).lower()+"_to_"+str(class_dict[NEW_CLASS]).lower()
    result_path = 'results/'+ SAVE_PATH +'_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    for LAYER in layer_cuts:
        for PERT_COUNT in perturbation_counts:
            for LOSS_FN in loss_functions:
                for EPS in epsilon:

                    print("\n\n\n")
                    print("############################### [ PERT_COUNT: "+str(PERT_COUNT)+" | LOSS_FN: "+str(loss_dict[LOSS_FN])+" | EPS: "+str(EPS)+" | "+str(LAYER)+" Layer ] ###############################")
                    print("\n")

                    DATASET_NAME = "single_"+str(class_dict[TARGET_CLASS]).lower()+"_to_"+str(class_dict[NEW_CLASS]).lower()+"_"+str(loss_dict[LOSS_FN])+"_"+str(PERT_COUNT)+"_pertcount_"+str(EPS)+"_eps_"+str(LAYER)+"layer"
                    best_image_id = CUSTOM_BEST_IMAGE_ID
                    best_image_id = generate_single_image_pertubed_dataset(model_name="basic_training", output_name=DATASET_NAME, target_class=TARGET_CLASS, new_class=NEW_CLASS, EPS=EPS, ITERS=ITERS, pertube_count=PERT_COUNT, loss_fn=LOSS_FN, custom_id=CUSTOM_BEST_IMAGE_ID, device_name=device, layer_cut=LAYER)

                    train(epochs=EPOCHS, learning_rate=LR, output_name="basic_training_"+str(DATASET_NAME), data_suffix=DATASET_NAME, batch_size=BATCH_SIZE, device_name=device, data_augmentation=True)

                    analyze_layers(EPS, ITERS, target_class=TARGET_CLASS, new_class=NEW_CLASS, save_path=result_path, model_name="basic_training_"+str(DATASET_NAME), target_id=best_image_id, pert_count=PERT_COUNT, loss_fn=LOSS_FN, device_name=device, layer_cut=LAYER)
                    evaluate_single_class(model_name="basic_training_"+str(DATASET_NAME), save_path=result_path, target_class=TARGET_CLASS, new_class=NEW_CLASS, EPS=EPS, ITERS=ITERS, pert_count=PERT_COUNT, loss_function=LOSS_FN, device_name=device, layer_cut=LAYER)
                    single_model_evaluation(model_name="basic_training_"+str(DATASET_NAME), save_path=result_path, target_class=TARGET_CLASS, new_class=NEW_CLASS, EPS=EPS, ITERS=ITERS, pert_count=PERT_COUNT, loss_function=LOSS_FN, device_name=device, layer_cut=LAYER)


    print("finished: [ " + str(DATASET_NAME) + " ]")
    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

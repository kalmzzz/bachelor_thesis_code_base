import time
import os
from train_methods import *
from evaluation_methods import *
from dataset_generation_methods import *
import torch
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
import sys

cudnn.benchmark = True


# ---------------- Parameters -----------------------
airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
class_dict_rev = {"airplane":0, "auto":1, "bird":2, "cat":3, "deer":4, "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}
class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
loss_dict = {0:"BCE_WithLogits", 1:"Wasserstein", 2:"KLDiv"}
BCE, WASSERSTEIN, KLDIV = 0, 1, 2
ITERS = 100 #wiederholungen von pgd
EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128

# ------------------------------------------------------------------------------------------------------

def main(eps, gpu, pert_counts, loss_fn, layer_cuts, target_class, new_class, image_id, eval):
    start = time.perf_counter()

    global device
    device = 'cpu'
    if gpu == 0:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if gpu == 1:
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    print("\n\n\n")
    print("Device: ", str(device))
    print("Eps: ", str(eps))
    print("Perurbation_Count: ", str(pert_counts)) #gibt an wieviel Prozent der Zielklasse "perturbed" sein sollen
    print("Loss_Function: ", str(loss_fn))
    print("Layer_Cuts: ", str(layer_cuts))
    print("Target_Class: ", str(target_class)) #Target Class wird als new class erkannt während new class normal erkannt wird
    print("New_Class: ", str(new_class))
    print("Image_Id: ", str(image_id)) #wenn nicht das beste bild genommen werden soll, kann man hier eine wunsch id genommen werden
    CUSTOM_BEST_IMAGE_ID = image_id #wenn nicht das beste bild genommen werden soll, kann man hier eine wunsch id genommen werden

    target_class = class_dict_rev[target_class]
    new_class = class_dict_rev[new_class]

    SAVE_PATH = "single_"+str(class_dict[target_class]).lower()+"_to_"+str(class_dict[new_class]).lower()
    result_path = 'results/'+ SAVE_PATH +'_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    for LAYER in layer_cuts:
        for PERT_COUNT in pert_counts:
            for LOSS_FN in loss_fn:
                for EPS in eps:

                    print("\n\n\n")
                    print("############################### [ PERT_COUNT: "+str(PERT_COUNT)+" | LOSS_FN: "+str(loss_dict[LOSS_FN])+" | EPS: "+str(EPS)+" | "+str(LAYER)+" Layer ] ###############################")
                    print("\n")

                    DATASET_NAME = "single_"+str(class_dict[target_class]).lower()+"_to_"+str(class_dict[new_class]).lower()+"_"+str(loss_dict[LOSS_FN])+"_"+str(PERT_COUNT)+"_pertcount_"+str(EPS)+"_eps_"+str(LAYER)+"layer"
                    best_image_id = CUSTOM_BEST_IMAGE_ID

                    if not eval:
                        best_image_id = generate_single_image_pertubed_dataset(model_name="basic_training", output_name=DATASET_NAME, target_class=target_class, new_class=new_class, EPS=EPS, ITERS=ITERS, pertube_count=PERT_COUNT, loss_fn=LOSS_FN, custom_id=best_image_id, device_name=device, layer_cut=LAYER)
                        train(epochs=EPOCHS, learning_rate=LR, output_name="basic_training_"+str(DATASET_NAME), data_suffix=DATASET_NAME, batch_size=BATCH_SIZE, device_name=device, data_augmentation=True)

                    analyze_layers(EPS, ITERS, target_class=target_class, new_class=new_class, save_path=result_path, model_name="basic_training_"+str(DATASET_NAME), target_id=best_image_id, pert_count=PERT_COUNT, loss_fn=LOSS_FN, device_name=device, layer_cut=LAYER)
                    evaluate_single_class(model_name="basic_training_"+str(DATASET_NAME), save_path=result_path, target_class=target_class, new_class=new_class, EPS=EPS, ITERS=ITERS, pert_count=PERT_COUNT, loss_function=LOSS_FN, device_name=device, layer_cut=LAYER)
                    single_model_evaluation(model_name="basic_training_"+str(DATASET_NAME), save_path=result_path, target_class=target_class, new_class=new_class, EPS=EPS, ITERS=ITERS, pert_count=PERT_COUNT, loss_function=LOSS_FN, device_name=device, layer_cut=LAYER)


    print("finished: [ " + str(DATASET_NAME) + " ]")
    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")



# ------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu",             "-g", help="GPU", type=int, default=0)
    parser.add_argument("--eps",             "-e", help="Epsilon", nargs='+', type=float, default=0.5)
    parser.add_argument("--pert_counts",     "-p", help="Perturbation Percentage", nargs='+', type=float, default=0.5)
    parser.add_argument("--loss_fn",         "-l", help="Loss Function", type=int, nargs='+', default=2)
    parser.add_argument("--layer_cuts",      "-c", help="Layer Cuts", type=int, nargs='+', default=1)
    parser.add_argument("--target_class",    "-t", help="Target Class", type=str, required=True)
    parser.add_argument("--new_class",       "-n", help="New Class", type=str, required=True)
    parser.add_argument("--image_id",        "-i", help="Custom Best Image ID", type=int, default=None)
    parser.add_argument("--eval",            "-v", help="Wont train, just evaluation", action='store_true')

    args = parser.parse_args()

    main(**vars(args))

    #python run_server.py --gpu 0 --eps 2 1 0.75 0.5 0.25 0.1 --pert_counts 0.5 --loss_fn 2 --layer_cuts 1 2 --target_class "deer" --new_class "horse"

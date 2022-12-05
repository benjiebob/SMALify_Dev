import sys

sys.path.append("/scratch/code/SMALify")
sys.path.append("/scratch/code/SMALify/smal_fitter")

import streamlit as st

st.set_page_config(layout="wide")

import numpy as np
import matplotlib.pyplot as plt
from smal_fitter import SMALFitter
import pickle as pkl

import torch
from data_loader import load_badja_sequence, load_stanford_sequence
from viewer.exporter import ImageExporter
from tqdm import trange

import os
import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS


@st.cache(allow_output_mutation=True)
def load_sequence(image_range, dataset, name):
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    if dataset == "badja":
        data, filenames = load_badja_sequence(
            config.BADJA_PATH, name, config.CROP_SIZE, image_range=image_range
        )
    else:
        data, filenames = load_stanford_sequence(
            config.STANFORD_EXTRA_PATH, name, config.CROP_SIZE
        )

    return data, filenames


@st.cache(allow_output_mutation=True)
def load_fitter_model(data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert config.SHAPE_FAMILY >= 0, "Shape family should be greater than 0"

    use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR

    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        print(
            "WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters."
        )
        config.ALLOW_LIMB_SCALING = False

    model = SMALFitter(
        device, data, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior
    )

    return model


# TODO(Ben)
# - Load sequence from combo box
# - Choose frame from sequence
# - Show initial state of the model on the screen
# - Allow manual editing of global rotation/translation

# - Allow specification of the optimization weights / schedule
# - Allow initialization of the shape parameters (e.g. choose one of the Unity Dogs)

# - Allow downloading the result mesh/weights

# vis_frequency = config.VIS_FREQUENCY
vis_frequency = 25
dataset, name = config.SEQUENCE_OR_IMAGE_NAME.split(":")
image_range = [0]
data, filenames = load_sequence(image_range, dataset, name)
model = load_fitter_model(data)


click = st.button("Start the Fitter")
progress_bar = st.progress(0)

image_exporter = ImageExporter(config.OUTPUT_DIR, filenames)
image_exporter.stage_id = 0
image_exporter.epoch_name = str(0)
model.generate_visualization(image_exporter)  # Final stage


if click:

    dataset_size = len(filenames)
    print("Dataset size: {0}".format(dataset_size))

    for stage_id, weights in enumerate(np.array(config.OPT_WEIGHTS).T):
        opt_weight = weights[:6]
        w_temp = weights[6]
        epochs = int(weights[7])
        lr = weights[8]

        progress_bar.progress(0)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

        if stage_id == 0:
            model.joint_rotations.requires_grad = False
            model.betas.requires_grad = False
            model.log_beta_scales.requires_grad = False
            target_visibility = model.target_visibility.clone()
            model.target_visibility *= 0
            model.target_visibility[:, config.TORSO_JOINTS] = target_visibility[
                :, config.TORSO_JOINTS
            ]  # Turn on only torso points
        else:
            model.joint_rotations.requires_grad = True
            model.betas.requires_grad = True
            if config.ALLOW_LIMB_SCALING:
                model.log_beta_scales.requires_grad = True
            model.target_visibility = data[-1].clone()

        t = trange(epochs, leave=True)
        for epoch_id in t:
            progress_bar.progress(epoch_id / epochs)
            image_exporter.stage_id = stage_id
            image_exporter.epoch_name = str(epoch_id)

            acc_loss = 0
            optimizer.zero_grad()
            for j in range(0, dataset_size, config.WINDOW_SIZE):
                batch_range = list(range(j, min(dataset_size, j + config.WINDOW_SIZE)))
                loss, losses = model(batch_range, opt_weight, stage_id)
                acc_loss += loss.mean()
                # print ("Optimizing Stage: {}\t Epoch: {}, Range: {}, Loss: {}, Detail: {}".format(stage_id, epoch_id, batch_range, loss.data, losses))

            joint_loss, global_loss, trans_loss = model.get_temporal(w_temp)

            desc = "EPOCH: Optimizing Stage: {}\t Epoch: {}, Loss: {:.2f}, Temporal: ({}, {}, {})".format(
                stage_id,
                epoch_id,
                acc_loss.data,
                joint_loss.data,
                global_loss.data,
                trans_loss.data,
            )

            t.set_description(desc)
            t.refresh()

            acc_loss = acc_loss + joint_loss + global_loss + trans_loss
            acc_loss.backward()
            optimizer.step()

            if epoch_id % vis_frequency == 0:
                model.generate_visualization(image_exporter)

    image_exporter.stage_id = 10
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter)  # Final stage

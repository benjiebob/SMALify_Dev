import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import plotly.graph_objects as go
import trimesh
import cv2
import os
from copy import deepcopy
# pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
# x, y, z = pts.T

CHECKPOINT_DIR = "/scratch/code/SMALify/checkpoints"
exp_choices = os.listdir(CHECKPOINT_DIR)

menu = iter(st.columns([1, 1]))
with next(menu):
    exp_basename = st.selectbox("Select experiment", exp_choices)

exp_dir = os.path.join(CHECKPOINT_DIR, exp_basename)
frames = [ int(x) for x in os.listdir(exp_dir) ]

with next(menu):
    frame = st.slider("Select frame", min_value=min(frames), max_value=max(frames), value=min(frames))

input_path = f"{exp_dir}/{frame:04d}/st10_ep0.ply"
mesh = trimesh.load_mesh(input_path)
mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)

fig = go.Figure(data=[go.Mesh3d(
    x=mesh.vertices[:,0], 
    y=mesh.vertices[:,1], 
    z=mesh.vertices[:,2], 
    i=mesh.faces[: ,0],
    j=mesh.faces[:, 1],
    k=mesh.faces[:, 2], 
    lighting=dict(ambient=0.5, diffuse=0.5, fresnel=0.0, specular=0, roughness=0.0),
    lightposition=dict(x=0.0, y=0.0, z=200.0),
    color='lightpink', opacity=1.0)])

hack_points = np.array([
    [-1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0]])

fig.add_trace(
    go.Scatter3d(
        x=-1 * hack_points[:, 0],
        y=-1 * hack_points[:, 2],
        z=-1 * hack_points[:, 1],
        mode='markers',
        name='_fake_pts',
        visible=True,
        marker=dict(
            size=1,
            opacity = 1,
            color=(0.0, 0.0, 0.0),
        )))

# fig.show()
box_size = 2

# fig.update_scenes(patch = dict(lightposition=(1.0, 0.0, 1.0)))



img = cv2.imread(input_path.replace(".ply", ".png"))[:, :, ::-1]
st.image(img, caption="Input image", use_column_width=True)
    
cols = iter(st.columns([1, 1, 1]))

with next(cols):
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=0, z=1.0)
    )
    scene = dict(
        xaxis = dict(nticks=10, range=[-box_size,box_size]),
        yaxis = dict(nticks=10, range=[-box_size,box_size]),
        zaxis = dict(nticks=10, range=[-box_size,box_size]),
        camera = camera)

    fig.update_scenes(patch = scene)
    st.plotly_chart(fig)
    

with next(cols):
    fig2 = deepcopy(fig)
    camera = dict(
        up=dict(x=0, y=1.0, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.0, y=0, z=0.0)
    )
    scene = dict(
        xaxis = dict(nticks=10, range=[-box_size,box_size]),
        yaxis = dict(nticks=10, range=[-box_size,box_size]),
        zaxis = dict(nticks=10, range=[-box_size,box_size]),
        camera = camera)

    fig2.update_scenes(patch = scene)
    st.plotly_chart(fig2)


with next(cols):
    fig3 = deepcopy(fig)
    camera = dict(
        up=dict(x=0, y=0, z=1.0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=1.0, z=0.0)
    )
    scene = dict(
        xaxis = dict(nticks=10, range=[-box_size,box_size]),
        yaxis = dict(nticks=10, range=[-box_size,box_size]),
        zaxis = dict(nticks=10, range=[-box_size,box_size]),
        camera = camera)

    fig3.update_scenes(patch = scene)
    st.plotly_chart(fig3)



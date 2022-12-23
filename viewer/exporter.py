import trimesh
import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from copy import deepcopy
import config
from PIL import ImageColor


class ImageExporter:
    def __init__(self, output_dir, filenames):
        self.output_dirs = self.generate_output_folders(output_dir, filenames)
        self.stage_id = 0
        self.epoch_name = 0
        self.placeholder = st.empty()

    def generate_output_folders(self, root_directory, filename_batch):
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        output_dirs = []
        for filename in filename_batch:
            filename_path = os.path.join(root_directory, os.path.splitext(filename)[0])
            output_dirs.append(filename_path)
            if not os.path.exists(filename_path):
                os.mkdir(filename_path)

        return output_dirs

    def add_hack_points(self, fig):
        hack_points = np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=-1 * hack_points[:, 0],
                y=-1 * hack_points[:, 2],
                z=-1 * hack_points[:, 1],
                mode="markers",
                name="_fake_pts",
                visible=True,
                marker=dict(
                    size=1,
                    opacity=1,
                    color=(0.0, 0.0, 0.0),
                ),
            )
        )

    def add_mesh(self, mesh, joints=None):
        objs = [
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                lighting=dict(
                    ambient=0.5, diffuse=0.5, fresnel=0.0, specular=0, roughness=0.0
                ),
                lightposition=dict(x=0.0, y=0.0, z=200.0),
                color="lightpink",
                opacity=0.5,
            )
        ]
        if joints is not None:
            for joint_id, joint in enumerate(joints):
                # color = ImageColor.getcolor(config.N_POSE_COLORS[joint_id], "RGB")
                objs.append(
                    go.Scatter3d(
                        x=[joint[0]],
                        y=[joint[1]],
                        z=[joint[2]],
                        name=f"joint_{joint_id}",
                        visible="legendonly",
                        # color=color,
                        # mode="markers",
                        # marker=dict(color=[color]),
                    )
                )
        fig = go.Figure(data=objs)
        return fig

    def render_scene(
        self, mesh, joints=None, up=[0, 1, 0], eye=[0.0, 0.0, 1.0], box_size=1.5
    ):
        fig = self.add_mesh(mesh, joints)
        self.add_hack_points(fig)
        camera = dict(
            up=dict(x=up[0], y=up[1], z=up[2]),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        )
        scene = dict(
            xaxis=dict(nticks=10, range=[-box_size, box_size]),
            yaxis=dict(nticks=10, range=[-box_size, box_size]),
            zaxis=dict(nticks=10, range=[-box_size, box_size]),
            camera=camera,
        )

        fig.update_scenes(patch=scene)
        return fig

    def export(
        self,
        collage_np,
        batch_id,
        global_id,
        img_parameters,
        vertices,
        faces,
        joints=None,
    ):
        caption = f"st{self.stage_id}_ep{self.epoch_name}.png"

        self.placeholder.empty()
        with self.placeholder.container():
            st.image(collage_np, caption=caption, use_column_width=True)

            # Export mesh
            vertices = vertices[batch_id].cpu().numpy()
            joints = joints[batch_id].cpu().numpy() if joints is not None else None
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

            anchor = mesh.vertices.mean(axis=0)
            mesh.vertices = mesh.vertices - anchor
            if joints is not None:
                joints = joints - anchor
            fig_front = self.render_scene(
                mesh, joints, up=[0, 1, 0], eye=[0.0, 0.0, 1.0]
            )
            fig_side = self.render_scene(
                mesh, joints, up=[0, 1, 0], eye=[-1.0, 0.0, 0.0]
            )
            fig_top = self.render_scene(mesh, joints, up=[0, 0, 1], eye=[0.0, 1.0, 0.0])

            cols = iter(st.columns([1, 1, 1]))
            with next(cols):
                st.plotly_chart(fig_side, use_container_width=True)

            with next(cols):
                st.plotly_chart(fig_front, use_container_width=True)

            with next(cols):
                st.plotly_chart(fig_top, use_container_width=True)

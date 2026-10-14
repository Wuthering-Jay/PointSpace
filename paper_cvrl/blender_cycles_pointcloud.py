"""Blender-side Cycles renderer for a prepared scientific point-cloud scene.

This script is launched by ``render_pointcloud_backends.py`` through Blender's
bundled Python.  Its command-line interface is intentionally internal; user
configuration remains module-level in the parent script for IDE execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _arguments():
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if len(arguments) != 2:
        raise ValueError("Expected: <prepared_scene.npz> <output.png>")
    return Path(arguments[0]), Path(arguments[1])


def _material():
    material = bpy.data.materials.new("Scientific point colors")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    principled.inputs["Roughness"].default_value = 0.82
    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.18
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    links.new(attribute.outputs["Color"], principled.inputs["Base Color"])
    if "Emission Color" in principled.inputs:
        links.new(attribute.outputs["Color"], principled.inputs["Emission Color"])
        principled.inputs["Emission Strength"].default_value = 0.22
    return material


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render(input_path: Path, output_path: Path):
    with np.load(input_path) as source:
        points = source["points"].astype(np.float32)
        colors = source["colors"].astype(np.float32)
        camera_right = source["camera_right"].astype(np.float64)
        camera_up = source["camera_up"].astype(np.float64)
        camera_view = source["camera_view"].astype(np.float64)
        ortho_scale = float(source["ortho_scale"])
        point_radius = float(source["point_radius"])
        width = int(source["width"])
        height = int(source["height"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # The engine is stored with the prepared scene so comparisons can switch
    # between true Cycles and EEVEE without changing this Blender-side script.
    with np.load(input_path) as source:
        engine = str(source["engine"].item()) if "engine" in source else "BLENDER_EEVEE_NEXT"
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = 32
        scene.cycles.use_denoising = True
        scene.cycles.device = "CPU"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(output_path)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    world = bpy.data.worlds.new("White studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        1.0, 1.0, 1.0, 1.0
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.72
    scene.world = world

    point_cloud = bpy.data.pointclouds.new("ALS point cloud")
    point_cloud.resize(points.shape[0])
    point_cloud.points.foreach_set("co", points.ravel())
    point_cloud.points.foreach_set(
        "radius", np.full(points.shape[0], point_radius, dtype=np.float32)
    )
    attribute = point_cloud.attributes.new("Col", "FLOAT_COLOR", "POINT")
    rgba = np.column_stack((colors[:, :3], np.ones(points.shape[0], dtype=np.float32)))
    attribute.data.foreach_set("color", rgba.ravel())
    point_cloud.materials.append(_material())
    cloud_object = bpy.data.objects.new("ALS point cloud", point_cloud)
    bpy.context.collection.objects.link(cloud_object)

    camera_data = bpy.data.cameras.new("Orthographic camera")
    camera = bpy.data.objects.new("Orthographic camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    distance = max(float(np.ptp(points, axis=0).max()) * 3.0, 10.0)
    camera.location = Vector(camera_view * distance)
    rotation = Matrix(
        (
            Vector(camera_right),
            Vector(camera_up),
            Vector(camera_view),
        )
    ).transposed()
    camera.matrix_world = rotation.to_4x4()
    camera.location = Vector(camera_view * distance)
    camera_data.lens = 50
    scene.camera = camera

    area_data = bpy.data.lights.new("Large softbox", "AREA")
    area_data.energy = 1150.0
    area_data.shape = "DISK"
    area_data.size = max(float(np.ptp(points[:, :2], axis=0).max()), 20.0)
    area = bpy.data.objects.new("Large softbox", area_data)
    bpy.context.collection.objects.link(area)
    area.location = Vector((-0.55 * distance, 0.60 * distance, 0.95 * distance))
    _look_at(area, (0.0, 0.0, 0.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    render(*_arguments())

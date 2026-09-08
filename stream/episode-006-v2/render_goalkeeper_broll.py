#!/usr/bin/env python3
"""Create original cinematic goalkeeper B-roll with no celebrity likeness."""

import math
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent


def mat(name, color, metallic=0.0, roughness=0.45, emission=None):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.metallic = metallic
    m.roughness = roughness
    if emission:
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Emission Color"].default_value = (*emission, 1)
        bsdf.inputs["Emission Strength"].default_value = 3.0
    return m


def cube(name, location, scale, material, bevel=0.08):
    bpy.ops.mesh.primitive_cube_add(location=location)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = o.modifiers.new("soft_edges", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    o.data.materials.append(material)
    return o


def sphere(name, location, radius, material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=radius, location=location)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def cylinder(name, location, radius, depth, material, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=depth,
                                       location=location, rotation=rotation)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 180
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("night_stadium")
scene.world.color = (0.004, 0.008, 0.015)

grass = mat("deep_green", (0.012, 0.10, 0.065), roughness=0.9)
white = mat("goal_white", (0.86, 0.92, 0.95), metallic=0.15, roughness=0.28)
keeper_mat = mat("keeper_blue", (0.02, 0.17, 0.36), metallic=0.1, roughness=0.32)
glove_mat = mat("gloves", (0.95, 0.55, 0.08), roughness=0.35)
skin = mat("neutral_skin", (0.32, 0.20, 0.14), roughness=0.7)
ball_mat = mat("ball", (0.95, 0.96, 0.98), roughness=0.32)
cyan = mat("stadium_light", (0.03, 0.34, 0.50), emission=(0.05, 0.65, 0.95))

cube("pitch", (0, 0, -0.08), (7.5, 7.0, 0.08), grass, 0)
for x in (-3.5, 3.5):
    cylinder("goal_post", (x, 1.2, 1.7), 0.09, 3.4, white)
cylinder("crossbar", (0, 1.2, 3.4), 0.09, 7.0, white, (0, math.pi / 2, 0))
cube("goal_line", (0, 1.15, 0.012), (3.5, 0.035, 0.012), white, 0)
for x in range(-3, 4):
    cube("net_v", (x, 1.45, 1.65), (0.012, 0.012, 1.65), white, 0)
for z in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    cube("net_h", (0, 1.45, z), (3.45, 0.012, 0.012), white, 0)

rig = bpy.data.objects.new("keeper_rig", None)
bpy.context.collection.objects.link(rig)
torso = cube("torso", (0, 0.72, 1.45), (0.30, 0.20, 0.52), keeper_mat, 0.16)
head = sphere("head", (0, 0.72, 2.22), 0.24, skin)
parts = [torso, head]
for sx in (-1, 1):
    arm = cylinder("arm", (sx * 0.58, 0.72, 1.64), 0.09, 0.82, keeper_mat,
                   (0, math.pi / 2, 0))
    glove = sphere("glove", (sx * 1.02, 0.72, 1.64), 0.16, glove_mat)
    leg = cylinder("leg", (sx * 0.20, 0.72, 0.60), 0.11, 1.08, keeper_mat)
    parts += [arm, glove, leg]
for p in parts:
    p.parent = rig

rig.location = (0, 0, 0)
rig.rotation_euler = (0, 0, 0)
rig.keyframe_insert("location", frame=1)
rig.keyframe_insert("rotation_euler", frame=1)
rig.keyframe_insert("location", frame=55)
rig.keyframe_insert("rotation_euler", frame=55)
rig.location = (1.85, 0.02, 0.42)
rig.rotation_euler = (0, math.radians(78), 0)
rig.keyframe_insert("location", frame=112)
rig.keyframe_insert("rotation_euler", frame=112)
rig.location = (2.15, 0.05, 0.28)
rig.rotation_euler = (0, math.radians(88), 0)
rig.keyframe_insert("location", frame=160)
rig.keyframe_insert("rotation_euler", frame=160)

ball = sphere("ball", (-0.4, -5.2, 0.28), 0.25, ball_mat)
ball.keyframe_insert("location", frame=1)
ball.location = (0.2, -3.7, 0.28)
ball.keyframe_insert("location", frame=45)
ball.location = (2.68, 0.60, 1.64)
ball.keyframe_insert("location", frame=112)
ball.location = (3.45, 0.20, 3.0)
ball.keyframe_insert("location", frame=158)

for i in range(22):
    x = -6.5 + (i % 11) * 1.3
    y = 3.6 + (i // 11) * 0.6
    s = sphere("crowd", (x, y, 1.1 + (i % 3) * 0.18), 0.20, keeper_mat)
    s.scale.z = 1.35

for x in (-5.8, -2.0, 2.0, 5.8):
    cube("stadium_strip", (x, 4.2, 3.8), (1.55, 0.05, 0.07), cyan, 0)

for loc, energy, color, size in [((-3, -2, 6), 1500, (0.30, 0.65, 1.0), 5),
                                 ((4, -1, 5), 1100, (1.0, 0.42, 0.12), 4)]:
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.object
    light.data.energy = energy
    light.data.color = color
    light.data.shape = "DISK"
    light.data.size = size
    look_at(light, (0, 0.5, 1.2))

bpy.ops.object.camera_add(location=(7.6, -10.2, 4.7))
camera = bpy.context.object
look_at(camera, (0, 0.2, 1.2))
camera.data.lens = 46
scene.camera = camera
camera.keyframe_insert("location", frame=1)
camera.location = (6.3, -8.8, 4.2)
camera.keyframe_insert("location", frame=112)

frames = HERE / "keeper-broll-frames"
frames.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(frames / "frame-")
bpy.ops.wm.save_as_mainfile(filepath=str(HERE / "keeper-broll.blend"))
bpy.ops.render.render(animation=True)

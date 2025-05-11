"""Contains helpful functions I use across different plugins"""
# slasheetools
#
# Copyright (except where otherwise noted) Slashee the Cow 2025-
#
#------------
# to_mesh_data(): from UltiMaker Cura
#  Copyright (c) 2019-2022 Ultimaker B.V., fieldOfView
#  Cura is released under the terms of the LGPLv3 or higher.
#
# to_trimesh(): Copyright fieldOfView
#------------
# - log():
#       A wrapper function arount UM.Logger that allows debug level messages
#       to be removed by changing DEBUG_LOG_MODE.
# - log_debug():
#       A wrapper function around log() that forces debug mode to be on.
#       The idea is that you import it as log for debugging but switch
#       to regular log for release versions.
# - validate_int():
#       Tests a str value to make sure it casts to an int fine and optionally
#       constrain it to upper or lower bounds.
# - validate_float():
#       Tests a str value to make sure it casts to an float fine and optionally
#       constrain it to upper or lower bounds.
# - to_mesh_data():
#       Converts a Trimesh to Uranium's MeshData.
#       By fieldOfView
# - to_trimesh():
#       Converts a Uranium MeshData object to a Trimesh object.
#       By fieldOfView
#
#------------
# v1: log() and log_debug() implementations including "dd" for debug that should show up anyway.
# v2: Added to_mesh_data() from Cura's built in AMFReader plugin and to_trimesh by fieldOfView

import math

import numpy as np
import trimesh

from UM.Mesh.MeshData import MeshData, calculateNormalsFromIndexedVertices
from UM.Logger import Logger

DEBUG_LOG_MODE = False

def log(level: str, message: str, debug: bool = DEBUG_LOG_MODE) -> None:
    """Wrapper function for logging messages using Cura's Logger,
    but with debug mode so as not to spam you."""
    if level == "d" and debug:
        Logger.log("d", message)
    elif level == "dd":
        Logger.log("d", message)
    elif level == "i":
        Logger.log("i", message)
    elif level == "w":
        Logger.log("w", message)
    elif level == "e":
        Logger.log("e", message)
    elif debug:
        Logger.log("w", f"Invalid log level: {level} for message {message}")

def log_debug(level: str, message: str) -> None:
    """Wrapper function for logging messages which ensures debug level messages will be logged"""
    log(level, message, True)

def validate_int(value: str, minimum: int | None = None, maximum: int | None = None,
                clamp: bool = False, default: int | None = None) -> int | None:
    """Safely casts a str to an int and optionally keeps it within bounds.
    Designed to be used on properties from QML files, but don't let that
    stop you from doing other things.
    """
    try:
        int_value = int(value)
    except ValueError:
        log("e", "validateInt got something which won't cast to an int.")
        return default

    if minimum is not None and int_value < minimum:
        return minimum if clamp else default

    if maximum is not None and int_value > maximum:
        return maximum if clamp else default

    return int_value

def validate_float(value: str, minimum: float = -math.inf, maximum: float = math.inf,
                clamp: bool = False, default: float | None = None) -> float | None:
    """Safely casts a str to a float and optionally keeps it within bounds.
    Designed to be used on properties from QML files, but don't let that
    stop you from doing other things.
    """
    try:
        float_value = float(value)
    except ValueError:
        log("e", "validateFloat got something which won't cast to a float.")
        return default

    if float_value < minimum:
        return minimum if clamp else default

    if float_value > maximum:
        return maximum if clamp else default

    return float_value

def to_mesh_data(tri_node: trimesh.base.Trimesh, file_name: str = "",
        rotation_angle: float = 90, rotation_direction: list[float] = [1,0,0]) -> MeshData:
    """Converts a Trimesh to Uranium's MeshData.

    :param tri_node: A Trimesh containing the contents of a file that was just read.
    :param file_name: The full original filename used to watch for changes
    :return: Mesh data from the Trimesh in a way that Uranium can understand it.
    """
    tri_node.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rotation_angle), rotation_direction))

    tri_faces = tri_node.faces
    tri_vertices = tri_node.vertices

    indices_list = []
    vertices_list = []

    index_count = 0
    face_count = 0
    for tri_face in tri_faces:
        face = []
        for tri_index in tri_face:
            vertices_list.append(tri_vertices[tri_index])
            face.append(index_count)
            index_count += 1
        indices_list.append(face)
        face_count += 1

    vertices = np.asarray(vertices_list, dtype = np.float32)
    indices = np.asarray(indices_list, dtype = np.int32)
    normals = calculateNormalsFromIndexedVertices(vertices, indices, face_count)

    mesh_data = MeshData(vertices = vertices, indices = indices, normals = normals, file_name = file_name)
    return mesh_data

def to_trimesh(mesh_data: MeshData) -> trimesh.base.Trimesh:
        if not mesh_data:
            return trimesh.base.Trimesh()

        indices = mesh_data.getIndices()
        if indices is None:
            # some file formats (eg 3mf) don't supply indices, but have unique vertices per face
            indices = np.arange(mesh_data.getVertexCount()).reshape(-1, 3)

        return trimesh.base.Trimesh(vertices=mesh_data.getVertices(), faces=indices)
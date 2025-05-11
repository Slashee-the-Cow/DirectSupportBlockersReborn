#--------------------------------------------------------------------------------------------------
# Direct Support Blockers Reborn
#
# Copyright Slashee the Cow 2025-
#
# Some parts taken from Cura by UltiMaker available under the LGPLv3
# https://github.com/Ultimaker/Cura
#--------------------------------------------------------------------------------------------------
# The name is actually a bit of a misnomer, this is to fill the void left
# by "Custom Support Eraser Plus" by 5axes but is not based on it.
#
# It's a branding thing. People expect "Reborn" from my 5axes continuations.
#--------------------------------------------------------------------------------------------------
# Does one really do a changelog for the first version?
# Maybe I just list the things that are different than that which this aims to modernise.
#--------------------------------------------------------------------------------------------------

import math

from PyQt6.QtCore import QTimer
import numpy as np
import trimesh
import trimesh.creation

from cura.CuraApplication import CuraApplication
from cura.PickingPass import PickingPass
from cura.Operations.SetParentOperation import SetParentOperation
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.Scene.SliceableObjectDecorator import SliceableObjectDecorator
from cura.Scene.BuildPlateDecorator import BuildPlateDecorator

from UM.Event import Event, MouseEvent
from UM.Math.Vector import Vector
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Operations.AddSceneNodeOperation import AddSceneNodeOperation
from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.RemoveSceneNodeOperation import RemoveSceneNodeOperation
from UM.Operations.TranslateOperation import TranslateOperation
from UM.Scene.Selection import Selection
from UM.Settings.SettingInstance import SettingInstance
from UM.Tool import Tool
from UM.i18n import i18nCatalog

from .slasheetools import log_debug as log, validate_int, validate_float

class DirectSupportBlockersReborn(Tool):

    def __init__(self):
        super().__init__()

        self._catalog = i18nCatalog("directsupportblockers")

        # Note: if the selection is cleared with this tool active, there is no way to switch to
        # another tool than to reselect an object (by clicking it) because the tool buttons in the
        # toolbar will have been disabled. That is why we need to ignore the first press event
        # after the selection has been cleared.
        Selection.selectionChanged.connect(self._onSelectionChanged)
        self._had_selection = False
        self._skip_press = False

        self._selection_pass = None
        self._application = CuraApplication.getInstance()
        self._controller = self.getController()

        self._had_selection_timer = QTimer()
        self._had_selection_timer.setInterval(0)
        self._had_selection_timer.setSingleShot(True)
        self._had_selection_timer.timeout.connect(self._selectionChangeDelay)

    def event(self, event):
        super().event(event)

        if event.type == Event.MousePressEvent and MouseEvent.LeftButton in event.buttons \
            and self._controller.getToolsEnabled():
            if self._skip_press:
                # The selection was previously cleared, do not add/remove an support mesh but
                # use this click for selection and reactivating this tool only.
                self._skip_press = False
                self._support_heights=0
                return

            if self._selection_pass is None:
                # The selection renderpass is used to identify objects in the current view
                self._selection_pass = self._application.getRenderer().getRenderPass("selection")
            picked_node = self._controller.getScene().findObject(self._selection_pass.getIdAtPosition(event.x, event.y))
            if not picked_node:
                # There is no slicable object at the picked location
                return

            node_stack = picked_node.callDecoration("getStack")
            if node_stack:
                if node_stack.getProperty("anti_overhang_mesh", "value"):
                    self._removeBlocker(picked_node)
                    return

                elif node_stack.getProperty("support_mesh", "value") or node_stack.getProperty("infill_mesh", "value") or node_stack.getProperty("cutting_mesh", "value"):
                    # Only "normal" meshes can have anti_overhang_meshes added to them
                    return

            # Create a pass for picking a world-space location from the mouse location
            active_camera = self._controller.getScene().getActiveCamera()
            picking_pass = PickingPass(active_camera.getViewportWidth(), active_camera.getViewportHeight())
            picking_pass.render()

            picked_position = picking_pass.getPickedPosition(event.x, event.y)

            # Add the anti_overhang_mesh cube at the picked location
            self._createBlocker(picked_node, picked_position)

    def _createBlocker(self, parent: CuraSceneNode, position: Vector):
        node = CuraSceneNode()

        node.setName("Eraser")
        node.setSelectable(True)
        node.setCalculateBoundingBox(True)
        #mesh = self._createCube(10)
        mesh = self._createTube(5,12,15)
        node.setMeshData(mesh.build())
        node.calculateBoundingBoxMesh()

        active_build_plate = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        node.addDecorator(BuildPlateDecorator(active_build_plate))
        node.addDecorator(SliceableObjectDecorator())

        stack = node.callDecoration("getStack") # created by SettingOverrideDecorator that is automatically added to CuraSceneNode
        settings = stack.getTop()

        definition = stack.getSettingDefinition("anti_overhang_mesh")
        new_instance = SettingInstance(definition, settings)
        new_instance.setProperty("value", True)
        new_instance.resetState()  # Ensure that the state is not seen as a user state.
        settings.addInstance(new_instance)

        op = GroupedOperation()
        # First add node to the scene at the correct position/scale, before parenting, so the eraser mesh does not get scaled with the parent
        op.addOperation(AddSceneNodeOperation(node, self._controller.getScene().getRoot()))
        op.addOperation(SetParentOperation(node, parent))
        op.addOperation(TranslateOperation(node, position, set_position = True))
        op.push()

        self._application.getController().getScene().sceneChanged.emit(node)

    def _removeBlocker(self, node: CuraSceneNode):
        parent = node.getParent()
        if parent == self._controller.getScene().getRoot():
            parent = None

        op = RemoveSceneNodeOperation(node)
        op.push()

        if parent and not Selection.isSelected(parent):
            Selection.add(parent)

        CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node)

    def _updateEnabled(self):
        plugin_enabled = False

        global_container_stack = CuraApplication.getInstance().getGlobalContainerStack()
        if global_container_stack:
            plugin_enabled = global_container_stack.getProperty("anti_overhang_mesh", "enabled")

        CuraApplication.getInstance().getController().toolEnabledChanged.emit(self._plugin_id, plugin_enabled)

    def _onSelectionChanged(self):
        # When selection is passed from one object to another object, first the selection is cleared
        # and then it is set to the new object. We are only interested in the change from no selection
        # to a selection or vice-versa, not in a change from one object to another. A timer is used to
        # "merge" a possible clear/select action in a single frame
        if Selection.hasSelection() != self._had_selection:
            self._had_selection_timer.start()

    def _selectionChangeDelay(self):
        has_selection = Selection.hasSelection()
        if not has_selection and self._had_selection:
            self._skip_press = True
        else:
            self._skip_press = False

        self._had_selection = has_selection

    def _trimesh_to_meshbuilder(self, trimesh_model: trimesh.base.Trimesh,
            rotation_angle: float = 90, rotation_direction: list[float] = [1,0,0]) -> MeshBuilder:
        """Converts a Trimesh object to a MeshBuilder in a really ugly way so we get per-vertex normals."""
        trimesh_model.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rotation_angle), rotation_direction))

        mesh = MeshBuilder()

        verts = []
        indices = []

        trimesh_verts = trimesh_model.vertices
        trimesh_faces = trimesh_model.faces

        vert_count = 0
        for face in trimesh_faces:
            indices.append([vert_count, vert_count + 1, vert_count + 2])
            vert_count += 3
            
            for vertex_index in face:
                verts.append(trimesh_verts[vertex_index])

        mesh.setVertices(np.asarray(verts, dtype = np.float32))
        mesh.setIndices(np.asarray(indices, dtype = np.int32))

        mesh.calculateNormals()

        return mesh

    def _createCube(self, size: float) -> MeshBuilder:
        return self._trimesh_to_meshbuilder(trimesh.creation.box(extents = [size, size, size]))

    def _createTube(self, radius_inner: float, radius_outer: float, height: float) -> MeshBuilder:
        return self._trimesh_to_meshbuilder(trimesh.creation.annulus(r_min = radius_inner, r_max = radius_outer, height = height))
#--------------------------------------------------------------------------------------------------
# Direct Support Blockers Reborn
#
# Copyright Slashee the Cow 2025-
#
# Some large parts taken wholesale from Cura by UltiMaker available under the LGPLv3
# https://github.com/Ultimaker/Cura
#--------------------------------------------------------------------------------------------------
# The name is actually a bit of a misnomer, this is to fill the void left
# by "Custom Support Eraser Plus" by 5axes but is not based on it.
#
# It's a branding thing. People expect "Reborn" from my 5axes continuations.
#--------------------------------------------------------------------------------------------------
# Does one really do a changelog for the first version?
# Maybe I just list the things that are different than that which this aims to modernise.
# v1.0.0:
#   - Individual values for each dimension when making a box (much easier to work with than replacing a 10mm cube with a custom size cube)
#   - No cylinder option - I think there's such a thing as a niche use case, then there's cylindrical support blockers.
#--------------------------------------------------------------------------------------------------

import math

from PyQt6.QtCore import QTimer
import numpy as np
import trimesh
import trimesh.creation

from cura.CuraApplication import CuraApplication
from cura.Operations.SetParentOperation import SetParentOperation
from cura.PickingPass import PickingPass
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

    BLOCKER_TYPE_BOX: str = "box_blocker_type"
    BLOCKER_TYPE_PYRAMID: str = "pyramid_blocker_type"
    BLOCKER_TYPE_LINE: str = "line_blocker_type"
    BLOCKER_TYPE_CUSTOM: str = "custom_blocker_type"

    def __init__(self):
        super().__init__()

        self._catalog = i18nCatalog("directsupportblockers")

        self.setExposedProperties("InputsValid", "BlockerType", "BlockerToPlate", "BoxWidth", "BoxDepth", "BoxHeight")

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

        self._line_points: int = 0
        self._line_first_point: Vector = None
        self._line_second_point: Vector = None

        self._inputs_valid = True
        self._blocker_to_plate: bool = False
        self._click_height: float = 0.0
        self._blocker_type: str = self.BLOCKER_TYPE_BOX

        self._box_width: float = 10.0
        self._box_depth: float = 10.0
        self._box_height: float = 10.0

        self._pyramid_top_width: float = 10.0
        self._pyramid_top_depth: float = 10.0
        self._pyramid_bottom_width: float = 20
        self._pyramid_bottom_depth: float = 20
        self._pyramid_height: float = 20

        self._preferences = self._application.getPreferences()
        self._preferences.addPreference("directsupportblockers/blocker_to_plate", False)
        self._preferences.addPreference("directsupportblockers/blocker_type", self.BLOCKER_TYPE_BOX)
        self._preferences.addPreference("directsupportblockers/box_width", 10)
        self._preferences.addPreference("directsupportblockers/box_depth", 10)
        self._preferences.addPreference("directsupportblockers/box_height", 10)
        self._preferences.addPreference("directsupportblockers/pyramid_top_width", 10)
        self._preferences.addPreference("directsupportblockers/pyramid_top_depth", 10)
        self._preferences.addPreference("directsupportblockers/pyramid_bottom_width", 20)
        self._preferences.addPreference("directsupportblockers/pyramid_bottom_depth", 20)
        self._preferences.addPreference("directsupportblockers/pyramid_height", 20)
        

        self._blocker_to_plate = bool(self._preferences.getValue("directsupportblockers/blocker_to_plate"))
        self._blocker_type = self._preferences.getValue("directsupportblockers/blocker_type")
        self._box_width = float(self._preferences.getValue("directsupportblockers/box_width"))
        self._box_depth = float(self._preferences.getValue("directsupportblockers/box_depth"))
        self._box_height = float(self._preferences.getValue("directsupportblockers/box_height"))
        self._pyramid_top_width = float(self._preferences.getValue("directsupportblockers/pyramid_width"))
        self._pyramid_top_depth = float(self._preferences.getValue("directsupportblockers/pyramid_depth"))
        self._pyramid_botom_width = float(self._preferences.getValue("directsupportblockers/pyramid_width"))
        self._pyramid_bottom_depth = float(self._preferences.getValue("directsupportblockers/pyramid_depth"))
        self._pyramid_height = float(self._preferences.getValue("directsupportblockers/pyramid_height"))

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

            if self._blocker_type == self.BLOCKER_TYPE_LINE:
                self._line_points += 1
                if self._line_points == 1:
                    self._line_first_point = picking_pass.getPickedPosition(event.x, event.y)
                    return
                elif self._line_points == 2:
                    self._line_second_point = picking_pass.getPickedPosition(event.x, event.y)
                    #self._createBlocker(picked_node, self._line_first_point, self._line_second_point)
                    self._line_points = 0
            # Add the support blocker at the picked location
            self._createBlocker(picked_node, picked_position, self._line_first_point)

    def _createBlocker(self, parent: CuraSceneNode, position: Vector, position_start: Vector = None):
        if self._blocker_to_plate:
            self._click_height = position.z
        
        node = CuraSceneNode()

        node.setName("SupportBlocker")
        node.setSelectable(True)
        node.setCalculateBoundingBox(True)
        mesh = self._createCube([self._box_width, self._box_depth, self._box_height if not self._blocker_to_plate else self._click_height])
        #mesh = self._createTube(5,12,15)
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

        z_offset: float = 0.0
        if self._blocker_to_plate:
            z_offset = -(position.z / 2)
        else:
            match self._blocker_type:
                case self.BLOCKER_TYPE_BOX:
                    z_offset = -(self._box_height / 2)
                case self.BLOCKER_TYPE_PYRAMID:
                    z_offset = -(self._pyramid_height / 2)

        position.set(z=position.z + z_offset)

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

    def _createCube(self, size: list[float]) -> MeshBuilder:
        return self._trimesh_to_meshbuilder(trimesh.creation.box(extents = [size[0], size[1], size[2]]))

    def _create_box(self, width: float, depth: float, height: float) -> MeshBuilder:
        if self._blocker_to_plate:
            height = self._click_height
        return self._trimesh_to_meshbuilder(trimesh.creation.box(extents = [width, height, depth]))

    def _truncated_pyramid(self, base_dims, top_dims, height):
        """
        Creates a truncated square or rectangular pyramid.

        Args:
            base_dims: A tuple or list of [width, depth] for the base.
            top_dims: A tuple or list of [width, depth] for the top.
            height: The height of the pyramid.

        Returns:
            A trimesh.Trimesh object representing the truncated pyramid.
        """

        bw, bd = base_dims[0] / 2, base_dims[1] / 2
        tw, td = top_dims[0] / 2, top_dims[1] / 2

        # Define the vertices (bottom then top)
        vertices = np.array([
            [-bw, -bd, 0], [bw, -bd, 0], [bw, bd, 0], [-bw, bd, 0],  # Bottom
            [-tw, -td, height], [tw, -td, height], [tw, td, height], [-tw, td, height]   # Top
        ])

        # Define the faces (triangles for each side, then top and bottom)
        faces = [
            [0, 4, 1], [1, 4, 5],  # Front
            [1, 5, 2], [2, 5, 6],  # Right
            [2, 6, 3], [3, 6, 7],  # Back
            [3, 7, 0], [0, 7, 4],  # Left
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 2], [0, 2, 3]   # Bottom
        ]

        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def _createTube(self, radius_inner: float, radius_outer: float, height: float) -> MeshBuilder:
        return self._trimesh_to_meshbuilder(trimesh.creation.annulus(r_min = radius_inner, r_max = radius_outer, height = height))

    def getBlockerToPlate(self) -> bool:
        return self._blocker_to_plate

    def setBlockerToPlate(self, value) -> None:
        try:
            new_value = bool(value)
        except ValueError:
            log("e", "setBlockerToPlate got something which won't cast to a bool.")
            return

        self._blocker_to_plate = new_value

    def getBlockerType(self) -> str:
        return self._blocker_type

    def setBlockerType(self, value: str) -> None:
        self._blocker_type = value
        self._preferences.setValue("directsupportblockers/blocker_type", self._blocker_type)

        if self._blocker_type != self.BLOCKER_TYPE_LINE:
            self._line_points = 0
            self._line_first_point = None
            self._line_second_point = None

    def getInputsValid(self) -> bool:
        """This probably won't come up since it's only ever set by the QML"""
        return self._inputs_valid

    def setInputsValid(self, value: bool) -> None:
        self._inputs_valid = value

    def getBoxWidth(self) -> float:
        return self._box_width

    def setBoxWidth(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._box_width = new_value
            self._preferences.setValue("directsupportblockers/box_width", self._box_width)

    def getBoxDepth(self) -> float:
        return self._box_depth

    def setBoxDepth(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._box_depth = new_value
            self._preferences.setValue("directsupportblockers/box_depth", self._box_depth)

    def getBoxHeight(self) -> float:
        return self._box_height

    def setBoxHeight(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._box_height = new_value
            self._preferences.setValue("directsupportblockers/box_height", self._box_height)

    def getPyramidTopWidth(self) -> float:
        return self._pyramid_top_width

    def setPyramidTopWidth(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._pyramid_top_width = new_value
            self._preferences.setValue("directsupportblockers/pyramid_top_width", self._pyramid_top_width)

    def getPyramidTopDepth(self) -> float:
        return self._pyramid_top_depth

    def setPyramidTopDepth(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._pyramid_top_depth = new_value
            self._preferences.setValue("directsupportblockers/pyramid_top_depth", self._pyramid_top_depth)

    def getPyramidBottomWidth(self) -> float:
        return self._pyramid_bottom_width

    def setPyramidWidth(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._pyramid_bottom_width = new_value
            self._preferences.setValue("directsupportblockers/pyramid_bottom_width", self._pyramid_bottom_width)

    def getPyramidBottomDepth(self) -> float:
        return self._pyramid_bottom_depth

    def setPyramidBottomDepth(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._pyramid_bottom_depth = new_value
            self._preferences.setValue("directsupportblockers/pyramid_bottom_depth", self._pyramid_bottom_depth)

    def getPyramidHeight(self) -> float:
        return self._pyramid_height

    def setPyramidHeight(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._pyramid_height = new_value
            self._preferences.setValue("directsupportblockers/pyramid_height", self._pyramid_height)

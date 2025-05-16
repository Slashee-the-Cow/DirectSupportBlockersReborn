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
#   - No cylinder option - I think there's such a thing as a niche use case, then there's cylindrical support blockers. Ju
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

        self.setExposedProperties("InputsValid", "BlockerType", "BlockerToPlate", "BoxWidth", "BoxDepth", "BoxHeight", "PyramidTopWidth", "PyramidTopDepth", "PyramidBottomWidth", "PyramidBottomDepth", "PyramidHeight", "LineWidth", "LineHeight")

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

        self._line_width: float = 10
        self._line_height: float = 20

        self._preferences = self._application.getPreferences()
        self._preferences.addPreference("directsupportblockers/blocker_to_plate", False)
        self._preferences.addPreference("directsupportblockers/blocker_type", self.BLOCKER_TYPE_BOX)
        self._preferences.addPreference("directsupportblockers/box_width", 10)
        self._preferences.addPreference("directsupportblockers/box_depth", 15)
        self._preferences.addPreference("directsupportblockers/box_height", 20)
        self._preferences.addPreference("directsupportblockers/pyramid_top_width", 5)
        self._preferences.addPreference("directsupportblockers/pyramid_top_depth", 5)
        self._preferences.addPreference("directsupportblockers/pyramid_bottom_width", 20)
        self._preferences.addPreference("directsupportblockers/pyramid_bottom_depth", 20)
        self._preferences.addPreference("directsupportblockers/pyramid_height", 20)
        self._preferences.addPreference("directsupportblockers/line_width", 10)
        self._preferences.addPreference("directsupportblockers/line_height", 20)
        

        self._blocker_to_plate = bool(self._preferences.getValue("directsupportblockers/blocker_to_plate"))
        self._blocker_type = self._preferences.getValue("directsupportblockers/blocker_type")
        self._box_width = float(self._preferences.getValue("directsupportblockers/box_width"))
        self._box_depth = float(self._preferences.getValue("directsupportblockers/box_depth"))
        self._box_height = float(self._preferences.getValue("directsupportblockers/box_height"))
        self._pyramid_top_width = float(self._preferences.getValue("directsupportblockers/pyramid_top_width"))
        self._pyramid_top_depth = float(self._preferences.getValue("directsupportblockers/pyramid_top_depth"))
        self._pyramid_bottom_width = float(self._preferences.getValue("directsupportblockers/pyramid_bottom_width"))
        self._pyramid_bottom_depth = float(self._preferences.getValue("directsupportblockers/pyramid_bottom_depth"))
        self._pyramid_height = float(self._preferences.getValue("directsupportblockers/pyramid_height"))
        self._line_width = float(self._preferences.getValue("directsupportblockers/line_width"))
        self._line_height = float(self._preferences.getValue("directsupportblockers/line_height"))

    def event(self, event):
        super().event(event)

        if event.type == Event.MousePressEvent and MouseEvent.LeftButton in event.buttons \
            and self._controller.getToolsEnabled():
            if self._skip_press:
                # The selection was previously cleared, do not add/remove an support mesh but
                # use this click for selection and reactivating this tool only.
                self._skip_press = False
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
            #log("d", f"picked_position: {picked_position}")
            log("d", f"node {picked_node._name} has {picked_node.getMeshData().getVertices()}")

            if self._blocker_type == self.BLOCKER_TYPE_LINE:
                self._line_points += 1
                if self._line_points == 1:
                    self._line_first_point = picking_pass.getPickedPosition(event.x, event.y)
                    return
                if self._line_points == 2:
                    self._line_second_point = picking_pass.getPickedPosition(event.x, event.y)
                    self._line_points = 0
                    #self._createBlocker(picked_node, self._line_first_point, self._line_second_point)
            # Add the support blocker at the picked location
            self._createBlocker(picked_node, picked_position, self._line_first_point)

    def _createBlocker(self, parent: CuraSceneNode, position: Vector, position_start: Vector = None):
        if self._blocker_to_plate:
            self._click_height = position.z
        log("d", f"position: {position}, position_start: {position_start}")
        
        node = CuraSceneNode()

        node.setName("SupportBlocker")
        node.setSelectable(True)
        node.setCalculateBoundingBox(True)
        mesh = MeshBuilder()
        match self._blocker_type:
            case self.BLOCKER_TYPE_BOX:
                mesh = self._create_box(self._box_width, self._box_depth, self._box_height)
            case self.BLOCKER_TYPE_PYRAMID:
                mesh = self._create_truncated_pyramid((self._pyramid_top_width, self._pyramid_top_depth), (self._pyramid_bottom_width, self._pyramid_bottom_depth), self._pyramid_height if not self._blocker_to_plate else self._click_height)
            case self.BLOCKER_TYPE_LINE:
                mesh = self._create_line_mesh(self._line_width, position_start, position, 0.2, self._blocker_to_plate, self._line_height)
                #mesh = self._create_line_mesh(position_start, position, self._line_width, self._blocker_to_plate, self._line_height)

        #mesh = self._createCube([self._box_width, self._box_depth, self._box_height if not self._blocker_to_plate else self._click_height])
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

        move_y_nudge: float = 0.98  # When moving stuff down multiply it by this so it still touches the object
        
        y_offset: float = 0.0
        if self._blocker_to_plate:
            y_offset = -(position.y / 2)
        else:
            match self._blocker_type:
                case self.BLOCKER_TYPE_BOX:
                    y_offset = 0
                case self.BLOCKER_TYPE_PYRAMID:
                    y_offset = -(self._pyramid_height * move_y_nudge)

        log("d", f"y_offset: {y_offset}, position before z_offset: {position}")

        new_position = position.set(y = position.y + y_offset)
        

        op = GroupedOperation()
        # First add node to the scene at the correct position/scale, before parenting, so the eraser mesh does not get scaled with the parent
        op.addOperation(AddSceneNodeOperation(node, self._controller.getScene().getRoot()))
        op.addOperation(SetParentOperation(node, parent))
        if self._blocker_type != self.BLOCKER_TYPE_LINE:
            log("d", f"addings translate operation to new_position {new_position}")
            op.addOperation(TranslateOperation(node, new_position, set_position = True))
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
            rotation_angle: float = -90, rotation_direction: list[float] | tuple[float] = (1,0,0)) -> MeshBuilder:
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

    def verts_faces_for_duplicate_meshbuilder(self, verts: np.array, faces: np.array):

        vertices = []
        indices = []
        
        vert_count = 0
        for face in faces:
            indices.append([vert_count, vert_count + 1, vert_count + 2])
            vert_count += 3

            for vertex_index in face:
                vertices.append(verts[vertex_index])
        return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


    def _create_box(self, width: float, depth: float, height: float) -> MeshBuilder:
        if self._blocker_to_plate:
            height = self._click_height
        return self._trimesh_to_meshbuilder(trimesh.creation.box(extents = [width, height, depth]), 0)

    def _create_truncated_pyramid(self, top_dims, base_dims, height):
        """
        Creates a truncated square or rectangular pyramid.

        Args:
            top_dims: A tuple or list of [width, depth] for the top.
            base_dims: A tuple or list of [width, depth] for the base.
            height: The height of the pyramid.

        Returns:
            A trimesh.Trimesh object representing the truncated pyramid.
        """

        tw, td = top_dims[0] / 2, top_dims[1] / 2
        bw, bd = base_dims[0] / 2, base_dims[1] / 2
        
        if self._blocker_to_plate:
            height = self._click_height

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

        return self._trimesh_to_meshbuilder(trimesh.Trimesh(vertices=vertices, faces=faces), -90)

    def _create_line_mesh(self, width, pos1: Vector , pos2: Vector, extra_height, blocker_to_plate: bool, fixed_height: float):
        mesh = MeshBuilder()
        half_width = width / 2
        log("d", f"pos1: {pos1}, type {type(pos1)}, pos2: {pos2}, type {type(pos2)}")

        line_vector = pos2 - pos1
        vector_top = Vector(0, extra_height, 0)
        vector_bottom_1 = Vector(0, -pos1.y if blocker_to_plate else -fixed_height, 0)
        vector_bottom_2 = Vector(0, -pos2.y if blocker_to_plate else -fixed_height, 0)

        # Width direction perpendicular to the line (in Y-up space)
        norm_vector = Vector.cross(line_vector, Vector(0, 1, 0)).normalized()
        width_offset_vector = norm_vector * half_width

        # Define corner vectors
        p1_t_right = (pos1 + vector_top + width_offset_vector).getData()
        p1_t_left = (pos1 + vector_top - width_offset_vector).getData()
        p2_t_right = (pos2 + vector_top + width_offset_vector).getData()
        p2_t_left = (pos2 + vector_top - width_offset_vector).getData()
        p1_b_right = (pos1 + vector_bottom_1 + width_offset_vector).getData()
        p1_b_left = (pos1 + vector_bottom_1 - width_offset_vector).getData()
        p2_b_right = (pos2 + vector_bottom_2 + width_offset_vector).getData()
        p2_b_left = (pos2 + vector_bottom_2 - width_offset_vector).getData()

        vertex_count = 24

        verts = [ # 6 faces, 4 corners each, order for consistent normals (Y-up order)
            p1_t_right, p1_t_left, p2_t_left, p2_t_right, # Top
            p1_t_right, p2_t_right, p2_b_right, p1_b_right, # Front
            p1_t_left, p1_t_right, p1_b_right, p1_b_left,   # Right (relative to start)
            p2_t_right, p2_t_left, p2_b_left, p2_b_right, # Back (relative to start)
            p1_t_left, p2_t_left, p2_b_left, p1_b_left,   # Left (relative to start)
            p1_b_right, p1_b_left, p2_b_left, p2_b_right, # Bottom
        ]

        mesh.setVertices(np.asarray(verts, dtype=np.float32))
        indices = []
        for i in range(0, vertex_count, 4): # All 6 quads (12 triangles)
            indices.append([i, i+2, i+1])
            indices.append([i, i+3, i+2])
        mesh.setIndices(np.asarray(indices, dtype=np.int32))

        mesh.calculateNormals()
        return mesh

    def _create_line_blocker(self, first_point: Vector, second_point: Vector, width: float, height: float = 0, to_plate: bool = False, top_extra: float = 0):
        point_1 = first_point.getData()
        point_2 = second_point.getData()
        vector_direction = point_2 - point_1

        # Calculate width vectors
        vector_direction_normal = np.linalg.norm(vector_direction)

        width_vector = np.array

        vector_direction_unit = np.array([0,0,0])

        if vector_direction_normal < 1e-6:
            # If points are very close, default to a line along the X-axis
            width_vector = np.array([0, width / 2, 0])
        else:
            vector_direction_unit = vector_direction / vector_direction_normal

            # Find an arbitrary vector perpendicular to v_dir_unit
            if abs(vector_direction_unit[0]) > 1e-6:
                perpendicular = np.array([-vector_direction_unit[1] - vector_direction_unit[2], vector_direction_unit[0], vector_direction_unit[0]], dtype=np.float64)
            elif abs(vector_direction_unit[1]) > 1e-6:
                perpendicular = np.array([vector_direction_unit[1], -vector_direction_unit[0] - vector_direction_unit[2], vector_direction_unit[1]], dtype=np.float64)
            else:  # abs(vector_direction_unit[2]) > 1e-6
                perpendicular = np.array([vector_direction_unit[2], vector_direction_unit[2], -vector_direction_unit[0] - vector_direction_unit[1]], dtype=np.float64)

            width_vector_unit = perpendicular / np.linalg.norm(perpendicular)
            width_vector = ((width * 2) * width_vector_unit).astype(np.float64)

        # Figure out vertices
        bottom_y_start = -point_1[1] if to_plate else -height
        bottom_y_end = -point_2[1] if to_plate else -height

        top_y_start = top_extra
        top_y_end = top_extra

        vertices = np.array([
            point_1 + [width_vector[0], top_y_start, width_vector[2]],   # TL1 (0)
            point_1 + [-width_vector[0], top_y_start, -width_vector[2]],   # TR1 (1)
            point_1 + [-width_vector[0], bottom_y_start, -width_vector[2]], # BR1 (2)
            point_1 + [width_vector[0], bottom_y_start, width_vector[2]], # BL1 (3)

            # Four corners at point2 (representing the end face)
            point_2 + [width_vector[0], top_y_end, width_vector[2]],      # TL2 (4)
            point_2 + [-width_vector[0], top_y_end, -width_vector[2]],      # TR2 (5)
            point_2 + [-width_vector[0], bottom_y_end, -width_vector[2]],    # BR2 (6)
            point_2 + [width_vector[0], bottom_y_end, width_vector[2]],    # BL2 (7)
        ], dtype=np.float32)

        faces = np.array([
            # Start face (at point1)
            [0, 1, 2],
            [2, 3, 0],
            # End face (at point2)
            [4, 5, 6],
            [6, 7, 4],
            # Side face 1
            [0, 4, 7],
            [7, 3, 0],
            # Side face 2
            [1, 5, 6],
            [6, 2, 1],
            # Bottom face
            [3, 2, 6],
            [6, 7, 3],
            # Top face
            [0, 1, 5],
            [5, 4, 0],
        ], dtype=np.uint32)

        mb_verts, mb_indices = self.verts_faces_for_duplicate_meshbuilder(vertices, faces)
        mesh = MeshBuilder()
        mesh.setVertices(mb_verts)
        mesh.setIndices(mb_indices)
        mesh.calculateNormals()

        return mesh

    """def _create_line_mesh(self, point1: Vector, point2: Vector, width: float, blocker_to_plate: bool, fixed_height: float) -> MeshBuilder:
        Creates a line mesh for support blockers using MeshBuilder with duplicated vertices.

        width_direction1 = self._line_calculate_width_vectors(point1, point2, width)
        verts = self._line_calculate_line_vertices(point1, point2, width_direction1, blocker_to_plate, fixed_height)
        faces = self._line_create_line_faces()

        duplicated_vertices, duplicated_indices = self.verts_faces_for_duplicate_meshbuilder(verts, faces)

        builder = MeshBuilder()
        builder.setVertices(duplicated_vertices.tolist())
        builder.setIndices(np.asarray(duplicated_indices).tolist())
        builder.calculateNormals()

        return builder
        
    def _line_calculate_width_vectors(self, point1: Vector, point2: Vector, width: float) -> np.ndarray:
        p1_data = point1.getData()
        p2_data = point2.getData()
        v_dir = p2_data - p1_data
        v_dir_norm = np.linalg.norm(v_dir)

        if v_dir_norm < 1e-6:
            return np.array([width / 2, 0, 0], dtype=np.float64)  # Default if points are too close

        v_dir_unit = v_dir / v_dir_norm

        # Find an arbitrary vector perpendicular to v_dir_unit
        if abs(v_dir_unit[0]) > 1e-6:
            perp1 = np.array([-v_dir_unit[1] - v_dir_unit[2], v_dir_unit[0], v_dir_unit[0]], dtype=np.float64)
        elif abs(v_dir_unit[1]) > 1e-6:
            perp1 = np.array([v_dir_unit[1], -v_dir_unit[0] - v_dir_unit[2], v_dir_unit[1]], dtype=np.float64)
        else:  # abs(v_dir_unit[2]) > 1e-6
            perp1 = np.array([v_dir_unit[2], v_dir_unit[2], -v_dir_unit[0] - v_dir_unit[1]], dtype=np.float64)

        perp1_unit = perp1 / np.linalg.norm(perp1)
        width_direction1 = (width / 2) * perp1_unit

        return width_direction1.astype(np.float32)

    def _line_calculate_line_vertices(self, point1: Vector, point2: Vector, width_direction1: np.ndarray, blocker_to_plate: bool, fixed_height: float) -> np.ndarray:
        p1 = point1.getData()
        p2 = point2.getData()

        # Determine Y-levels (Cura's vertical) for bottom
        bottom_y_start = -p1[1] if blocker_to_plate else -fixed_height
        bottom_y_end = -p2[1] if blocker_to_plate else -fixed_height

        # Top Y-levels (Cura's vertical) (click Y)
        top_y_start = 0.0
        top_y_end = 0.0

        vertices = np.array([
            # Four corners at point1 (representing the start face)
            p1 + [width_direction1[0], top_y_start, width_direction1[2]],   # TL1 (0)
            p1 + [-width_direction1[0], top_y_start, -width_direction1[2]],   # TR1 (1)
            p1 + [-width_direction1[0], bottom_y_start, -width_direction1[2]], # BR1 (2)
            p1 + [width_direction1[0], bottom_y_start, width_direction1[2]], # BL1 (3)

            # Four corners at point2 (representing the end face)
            p2 + [width_direction1[0], top_y_end, width_direction1[2]],      # TL2 (4)
            p2 + [-width_direction1[0], top_y_end, -width_direction1[2]],      # TR2 (5)
            p2 + [-width_direction1[0], bottom_y_end, -width_direction1[2]],    # BR2 (6)
            p2 + [width_direction1[0], bottom_y_end, width_direction1[2]],    # BL2 (7)
        ], dtype=np.float32)

        return vertices

    def _line_create_line_faces(self) -> np.ndarray:
        faces = np.array([
            # Start face (at point1)
            [0, 1, 2],
            [2, 3, 0],
            # End face (at point2)
            [4, 5, 6],
            [6, 7, 4],
            # Side face 1
            [0, 4, 7],
            [7, 3, 0],
            # Side face 2
            [1, 5, 6],
            [6, 2, 1],
            # Bottom face
            [3, 2, 6],
            [6, 7, 3],
            # Top face
            [0, 1, 5],
            [5, 4, 0],
        ], dtype=np.uint32)
        return faces"""
        
            
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

    def getLineWidth(self) -> float:
        return self._line_width

    def setLineWidth(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._line_width = new_value
            self._preferences.setValue("directsupportblockers/line_width", self._line_width)

    def getLineHeight(self) -> float:
        return self._line_height

    def setLineHeight(self, value: str):
        new_value = validate_float(value)
        if new_value is not None:
            self._line_height = new_value
            self._preferences.setValue("directsupportblockers/line_height", self._line_height)

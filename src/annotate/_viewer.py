# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer.py

"""
Implementation code for the Cortex Viewer.
"""

# Imports ----------------------------------------------------------------------

import glob
import colorsys
import numpy as np
import nibabel as nib
import os.path as op
import neuropythy as ny
import ipywidgets as ipw
from struct import unpack
from functools import partial
from neuropythy.geometry.util import barycentric_to_cartesian

from ._viewer_panels import CortexControlPanel, CortexFigurePanel

# The Cortex Viewer State ------------------------------------------------------

class CortexViewerState:
    """
    The state object for the Cortex Viewer tool.
    """
    
    __FLATMAP_KWARGS = {
        "mask"      : ( "parcellation", 43 ), 
        "map_right" : "right", 
        "radius"    : np.pi / 2
    }

    __PROPERTY_KWARGS = {
        "angle" : { "func": lambda prop, h: { f"polar_angle_{h}": prop }, 
                     "kwargs": { } },
        "eccen" : { "func": lambda prop, _: { "eccentricity": prop }, 
                     "kwargs": { } },
        "vexpl" : { "func": lambda prop, _: prop, 
                     "kwargs": { "cmap": "hot", "vmin": 0, "vmax": 100 } },
        **{ key : { "func": lambda prop, _: prop, 
                    "kwargs": { "cmap": "temperature_dark", "vmin": -8, "vmax": 8 } }
            for key in ( "faces", "bodies", "places", "words" ) },
    }

    # Folder with fsaverage sulcus labels (<hemi>.corticalsulc.mgz and
    # corticalsulc.mgz.txt), relative to the dataset directory. Optional.
    SULCI_SUBDIR = "sulci"

    @staticmethod
    def sulcus_color(k):
        """Color for sulcus label k (the flatmap figure uses the same rule)."""
        return colorsys.hsv_to_rgb((k * 0.618034) % 1.0, 0.75, 0.95)

    # Point type constants
    POINT_FIXED  = 2  # fixed head/tail
    POINT_USER   = 1  # user-placed
    POINT_INTERP = 0  # interpolated


    def __init__(
            self, 
            annotation_widgets, 
            dataset_directory = "/home/jovyan/datasets", 
        ):
        
        # Store intialization variables
        self.annotation_widgets = annotation_widgets 
        self.dataset_directory  = dataset_directory

        # Prepare fsaverage da_ta 
        self.fsaverage = self._get_fsaverage()

        # Prepare initial values from annotation widget
        self.dataset_index = self.get_dataset_index()
        self.dataset       = self.get_dataset()
        self.targets       = self.get_targets()
        self.annotation    = self.get_annotation()

        # Prepare dataset dependent annotation information and styler
        self.update_annot_cfg()
        self.update_styler() 

        # Set cortex viewer style (default) options
        self.style = {
            "inflation_percent" : 100,
            "overlay"           : "curvature",
            "overlay_alpha"     : 1.0, 
            "point_size"        : 1.5, 
            "line_width"        : 0.25,
            "line_interp"       : 10,
        }

        # Set cortex viewer view options (camera and drawing)
        self.view = {
            "camera_mode"  : "orbit",   # "orbit" (fixed up axis) or "trackball"
            "height"       : 640,       # height of the 3D panel in pixels
            "draw_on_mesh" : False,     # whether clicks on the mesh add points
            "click_mode"   : "points",  # "points" (add points) or "crosshair"
        }

        # The crosshair (None, or a dict; see set_crosshair_*)
        self.crosshair = None

        # Prepare participant 3d coordinate and mesh data
        self.update_participant()
        self.update_coordinates()
        self.update_mesh()
        self.update_overlay()

        # Extract the flatmap and annotation data
        self.update_flatmap_annotations()

        # Prepare surface annotations and path 
        self.update_surface_annotations() 


    # [Cortex Viwer] fsaverage Method ------------------------------------------

    def _get_fsaverage(self):
        """load fsaverage tesselations and inflated surface coordinates."""
        # Load the fsaverage object
        fsa = ny.freesurfer_subject("fsaverage")
        
        # Return fsaverage dictionary values
        sulci_dir = op.join(self.dataset_directory, CortexViewerState.SULCI_SUBDIR)
        out = {}
        for h in ( "lh", "rh" ):
            sulci_file = op.join(sulci_dir, f"{h}.corticalsulc.mgz")
            hemi    = fsa.hemis[h]
            flatmap = hemi.mask_flatmap(**CortexViewerState.__FLATMAP_KWARGS)
            out[h]  = {
                "tesselation" : hemi.tess.faces,
                "inflated"    : hemi.surface("inflated").coordinates,
                "flatmap"     : flatmap,
                # Lookup from surface vertex index to flatmap column (-1 for
                # vertices that are not in the flatmap). Used to map clicks on
                # the 3D mesh back to flatmap (figure) coordinates.
                "flat_index"  : CortexViewerState._make_flat_index(
                    hemi.vertex_count, flatmap.labels),
                # Sulcus label for each vertex (0 = none), or None if the
                # label files are not available.
                "sulci"       : (np.asarray(nib.load(sulci_file).dataobj)
                                 .ravel().astype(int)
                                 if op.isfile(sulci_file) else None),
            }
        return out


    @staticmethod
    def _make_flat_index(n_vertices, flatmap_labels):
        """Make an array mapping each surface vertex to its flatmap column."""
        flat_index = np.full(int(n_vertices), -1, dtype = int)
        flat_index[np.asarray(flatmap_labels, dtype = int)] = \
            np.arange(len(flatmap_labels))
        return flat_index

    # [Annotation Tool] Get Active Widget Methods ------------------------------

    def _get_active_annotation_tool(self):
        """Get the active multicanvas widget."""
        return self.annotation_widgets.children[self.dataset_index]


    def _get_active_selection_panel(self):
        """Get the active selection panel widget."""
        active_tool = self._get_active_annotation_tool()
        return active_tool.control_panel.selection_panel
    

    def _get_active_figure_panel(self):
        """Get the active figure panel widget."""
        active_tool = self._get_active_annotation_tool()
        return active_tool.figure_panel
    
    # [Annotation Tool] Get Selected Information Methods -----------------------

    def get_dataset_index(self):
        """Get the current dataset selection index."""
        return self.annotation_widgets.selected_index
    

    def get_dataset(self):
        """Get the current dataset selection widget."""
        return self.annotation_widgets.titles[self.dataset_index]
    

    def get_targets(self):
        """Get the active targets from the active annotation tool."""
        selection_panel = self._get_active_selection_panel()
        return { 
            key.lower(): value.value for key, value 
            in selection_panel.target_dropdowns.items() 
        }
    

    @property
    def participant(self):
        """Get the current participant selection."""
        return self.targets.get("participant", None)


    @staticmethod
    def convert_hemisphere(hemisphere):
        """Convert hemisphere string to code or vice versa."""
        str_to_code = { "Left Hemisphere"  : "lh", "Right Hemisphere" : "rh" }
        code_to_str = { v: k for k, v in str_to_code.items() }
        if hemisphere in str_to_code.keys():
            return str_to_code[hemisphere]
        elif hemisphere in code_to_str.keys():
            return code_to_str[hemisphere]
        else: 
            raise ValueError(f"Invalid hemisphere value: {hemisphere}")


    @property
    def hemisphere(self):
        """Get the current hemisphere selection."""
        hemisphere = self.targets.get("hemisphere", None)
        if hemisphere is not None:
            return self.convert_hemisphere(hemisphere)
        return hemisphere
    

    def get_annotation(self):
        """Get the annotation from the active annotation tool."""
        selection_panel = self._get_active_selection_panel()
        return selection_panel.annotation


    @property
    def active(self):
        """Get the name of the annotation that is currently edited (or `None`)."""
        selection_panel = self._get_active_selection_panel()
        return selection_panel.active


    @property
    def active_label(self):
        """Get the menu label of the active variable annotation (or `None`)."""
        selection_panel = self._get_active_selection_panel()
        return selection_panel.variable_select.label
    

    def get_style_annotation(self):
        """Get the current style annotation selection widget."""
        return self.style_panel.annotation
    
    # [Annotation Tool] Update State Methods -----------------------------------

    def update_annot_cfg(self):
        """Update the annotation configuration based on current state."""
        active_tool = self._get_active_annotation_tool()
        self.annot_cfg = active_tool.state.config.annotations
        self.template  = active_tool.state.template
            

    def update_styler(self):
        """Update the styler options based on current state."""
        active_tool = self._get_active_annotation_tool()
        self.styler = active_tool.state.style


    def update_flatmap_annotations(self):
        """Get the annotation tool's annotation dictionary"""
        figure_panel = self._get_active_figure_panel()
        self.flatmap_annotations = figure_panel.annotations    

    # [Cortex Viewer] Update Participant ---------------------------------------

    def _read_coordinates(self, filename):
        """Read cortical mesh coordinates from a <hemisphere>.3d.coordinates file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            coordinates = np.array(unpack(fstr, values)).reshape((3, -1))
        return coordinates
    

    def _read_property(self, filename):
        """Read cortical property data from a <hemisphere>.3d.<property> file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            prop = np.array(unpack(fstr, values)).reshape((-1, ))
        return prop
    

    def update_participant(self): 
        """Load participant dataset based on current state."""
        # Locate participant directory and files
        participant_dir = op.join(
            self.dataset_directory, self.dataset.lower(), self.participant)
        filenames = glob.glob(op.join(participant_dir, f"{self.hemisphere}.3d.*"))

        # Load participant midgray coordinates
        coordinates_file = [x for x in filenames if x.endswith("coordinates")][0]
        self.midgray = self._read_coordinates(coordinates_file)

        # Load remaining property files
        property_files = [x for x in filenames if x != coordinates_file]
        self.properties = {
            fname.split(".")[-1]: self._read_property(fname)
            for fname in property_files
        }


    def update_coordinates(self):
        """Update the cortical mesh coordinates based on the inflation value."""
        inflated_coordinates = self.fsaverage[self.hemisphere]["inflated"]
        self.coordinates = ((inflated_coordinates - self.midgray) * \
             (self.style["inflation_percent"] / 100.0)) + self.midgray


    def update_mesh(self):
        """Update the cortical mesh object based on the current state."""
        # Update the mesh with current coordinates and properties
        self.mesh = ny.geometry.Mesh(
            faces       = self.fsaverage[self.hemisphere]["tesselation"],
            coordinates = self.coordinates,
            properties  = self.properties
        )

        # Assign curvature color (for default coloring)
        self.curvature = ny.graphics.cortex_plot_colors(self.mesh)[:, :3]    

    
    def update_overlay(self):
        """Update the cortical mesh color based on curvature."""
        if self.style["overlay"] == "curvature":
            self.overlay = None
        elif self.style["overlay"] == "sulci":
            # Curvature colors, with each labelled sulcus painted on top
            labels = self.fsaverage[self.hemisphere]["sulci"]
            colors = np.array(self.curvature, dtype = float)
            if labels is not None:
                for k in np.unique(labels[labels > 0]):
                    colors[labels == k] = CortexViewerState.sulcus_color(k)
            self.overlay = colors
        else: # Get property kwargs and values 
            overlay_style = self.style["overlay"]
            prop_kwargs   = CortexViewerState.__PROPERTY_KWARGS[overlay_style]
            prop_values   = self.mesh.properties[overlay_style]
            prop_color    = prop_kwargs["func"](prop_values, self.hemisphere)
            self.overlay  = ny.graphics.cortex_plot_colors(self.mesh, 
                color = prop_color, **prop_kwargs["kwargs"])[:, :3]

    # [Cortex Viewer] Update Surface Coordinates -------------------------------

    @staticmethod
    def _barycentric_weights(point, triangle):
        """Barycentric weights of a 3D point projected onto a triangle.

        `triangle` is a 3 x 3 matrix whose columns are the triangle corners.
        The weights are clipped to be non-negative and sum to 1, so a point
        just outside the triangle (from rounding) maps to its nearest edge.
        """
        a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
        v0, v1, v2 = b - a, c - a, point - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        denom = d00 * d11 - d01 * d01
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            return None # degenerate triangle
        wb = (d11 * d20 - d01 * d21) / denom
        wc = (d00 * d21 - d01 * d20) / denom
        w  = np.clip(np.array([1.0 - wb - wc, wb, wc]), 0, None)
        return w / w.sum()


    def surface_to_flatmap(self, position, face = None):
        """Convert a point on the 3D mesh into flatmap (figure) coordinates.

        `position` is the clicked 3D point (in the current, possibly inflated,
        mesh coordinates) and `face` is the three vertex indices of the
        triangle that was clicked (as reported by k3d). If `face` is missing or
        does not contain the point, the nearest vertex is used instead.

        Returns a length-2 array, or `None` if the point is outside the region
        covered by the flatmap.
        """
        fsa        = self.fsaverage[self.hemisphere]
        flat_index = fsa["flat_index"]
        flat_xy    = np.asarray(fsa["flatmap"].coordinates) # (2, n_flat)
        coords     = self.coordinates                       # (3, n_vertices)
        position   = np.asarray(position, dtype = float).reshape(3)
        n_vertices = coords.shape[1]

        # First try: barycentric interpolation within the clicked triangle.
        if face is not None:
            face = np.asarray(face, dtype = int).reshape(-1)
            if face.shape == (3,) and np.all((face >= 0) & (face < n_vertices)):
                triangle = coords[:, face]
                # Accept the triangle only if the click is close to it (a
                # sanity check that k3d's face indices match our vertices).
                edge = np.max(np.linalg.norm(
                    triangle - np.roll(triangle, 1, axis = 1), axis = 0))
                dist = np.min(np.linalg.norm(triangle - position[:, None], axis = 0))
                w = self._barycentric_weights(position, triangle)
                if w is not None and dist <= 2 * edge + 1e-6:
                    cols = flat_index[face]
                    if np.any(cols < 0): return None
                    return flat_xy[:, cols] @ w

        # Fallback: the nearest vertex.
        vertex = int(np.argmin(np.sum((coords - position[:, None])**2, axis = 0)))
        col = flat_index[vertex]
        if col < 0: return None
        return flat_xy[:, col].copy()


    # [Cortex Viewer] Crosshair -----------------------------------------------

    def set_crosshair_from_flatmap(self, point):
        """Place the crosshair at a flatmap (figure) point.

        Returns `True` if the point is on the cortex, `False` if it is outside
        the flatmap (then there is no crosshair on the mesh).
        """
        fsa_flatmap = self.fsaverage[self.hemisphere]["flatmap"]
        point   = np.asarray(point, dtype = float).reshape(2, 1)
        address = fsa_flatmap.address(point)
        ok = np.all(np.isfinite(address["coordinates"])) and \
             np.all(np.asarray(address["faces"]) >= 0)
        self.crosshair = { "flatmap": point[:, 0],
                           "address": address if ok else None,
                           "vertex" : None }
        return bool(ok)


    def set_crosshair_from_surface(self, position, face = None):
        """Place the crosshair at a clicked point on the mesh.

        Returns the flatmap point, or `None` if the point is outside the
        flatmap (the crosshair is then shown on the mesh only, at the
        nearest vertex).
        """
        point = self.surface_to_flatmap(position, face)
        if point is not None and self.set_crosshair_from_flatmap(point):
            return point
        coords = self.coordinates
        position = np.asarray(position, dtype = float).reshape(3, 1)
        vertex = int(np.argmin(np.sum((coords - position)**2, axis = 0)))
        self.crosshair = { "flatmap": None, "address": None, "vertex": vertex }
        return None


    def crosshair_position(self):
        """The crosshair's 3D position on the current mesh, or `None`."""
        c = self.crosshair
        if c is None: return None
        if c["address"] is not None:
            return self._flatmap_to_surface(c["address"], self.coordinates)[:, 0]
        if c["vertex"] is not None:
            return self.coordinates[:, c["vertex"]]
        return None


    @staticmethod
    def _flatmap_to_surface(flatmap_address, mesh_coordinates):
        """Convert flatmap annotation coordinates to surface coordinates."""
        bary_faces  = flatmap_address["faces"]       # (3, n_faces)
        bary_coords = flatmap_address["coordinates"] # (2, n_points)
        tx = np.transpose(mesh_coordinates[:, bary_faces], (1, 0, 2)) # (3, 3, n_points)
        return barycentric_to_cartesian(tx, bary_coords) # (3, n_points)

    
    def _interpolate_coordinates(self, coordinates, point_types):
        """Interpolate coordinates along the path."""
        # Get number of interpolated points
        n = self.style["line_interp"] + 2

        # Intialize ararys to store interpolated coordinates
        x_interp = []; y_interp = []; ptype_interp = []

        # Initialize point type interpolation filler
        ptype_filler = [self.POINT_INTERP] * self.style["line_interp"]

        # Iterate over each segment and interpolate points  
        n_interp = coordinates.shape[0] - 1 
        for i in np.arange(n_interp): # for each pair of coordinates
            # Extract start and end coordinates and point types for the segment
            xs, xe = coordinates[i, 0], coordinates[i+1, 0]
            ys, ye = coordinates[i, 1], coordinates[i+1, 1]
            ps, pe = point_types[i], point_types[i+1]

            # Interpolate x and y coordinates and point types for the segment
            xn = np.linspace(xs, xe, n)
            yn = np.linspace(ys, ye, n)
            pn = [ps, *ptype_filler, pe]

            if i == 0: # for the first segment, include the starting point
                x_interp.append(xn)
                y_interp.append(yn)
                ptype_interp.append(pn)
            else: # for subsequent segments, exclude the starting point to avoid duplicates
                x_interp.append(xn[1:])
                y_interp.append(yn[1:])
                ptype_interp.append(pn[1:])

        # Concatenate and prepare interpolated points
        x_interp = np.concatenate(x_interp)
        y_interp = np.concatenate(y_interp)
        ptype_interp = np.concatenate(ptype_interp)

        # Return interpolated coordinates (as matrix) and point types (as int)
        interp_coordinates = np.vstack((x_interp, y_interp, ptype_interp)).T
        return interp_coordinates[:,:-1], interp_coordinates[:,-1].astype(int)


    def update_surface_addresses(self, annotations = None): 
        """Update cortical surface addresses for each annotation."""
        # Get the list of annotations to update
        if annotations is None:
            annotations = list(self.flatmap_annotations.keys())
        elif isinstance(annotations, str):
            annotations = [annotations, ]
        else:
            raise ValueError(f"Invalid annotations value: {annotations}")

        # Get current fsaverage hemisphere flatmap
        fsa_flatmap = self.fsaverage[self.hemisphere]["flatmap"]

        # Convert each flatmap annotation to surface coordinates
        for key in annotations: # for each annotation to update
            # Get the current annotaitons flatmap coordinates
            flatmap_coordinates = self.flatmap_annotations[key]

            # If no flatmap coordinates, set surface annotation to None
            if flatmap_coordinates is None or flatmap_coordinates.shape[0] == 0: 
                self.surface_annotations[key] = {
                    "addresses"   : None,
                    "coordinates" : None,
                    "point_types" : None,
                }
                continue
            
            # If there are flatmap coordinates, figure out each point type
            annotation  = self.template(key) # the (config) annotation
            n_points    = flatmap_coordinates.shape[0]
            point_types = np.full(n_points, self.POINT_USER)
            fixed_head  = bool(self.annot_cfg.fixed_head[annotation])
            fixed_tail  = bool(self.annot_cfg.fixed_tail[annotation])
            if fixed_head: point_types[0]  = self.POINT_FIXED
            if fixed_tail: point_types[-1] = self.POINT_FIXED

            # A boundary is closed: add its first point to the end as an
            # interpolated point, which draws the line but not another point.
            if self.annot_cfg.types[annotation] == "boundary" and n_points > 2:
                flatmap_coordinates = np.vstack(
                    [ flatmap_coordinates, flatmap_coordinates[:1] ])
                point_types = np.append(point_types, self.POINT_INTERP)

            # Interpolate coordinate if there are more than 1 point (to make a 
            # segment) and if the points are NOT all fixed points.
            if n_points > 1 and not np.all(point_types == self.POINT_FIXED):
                flatmap_coordinates, point_types = \
                    self._interpolate_coordinates(flatmap_coordinates, point_types)
            
            # Convert flatmap coordinates to addresses
            flatmap_address = fsa_flatmap.address(flatmap_coordinates.T)
        
            # Store surface annotation addresses
            self.surface_annotations[key] = {
                "addresses"   : flatmap_address,
                "coordinates" : None,
                "point_types" : point_types,
            }


    def update_surface_coordinates(self, annotations = None):
        """Update cortical surface coordinates for each annotation."""
        # Get the list of annotations to update
        if annotations is None:
            annotations = list(self.flatmap_annotations.keys())
        elif isinstance(annotations, str):
            annotations = [annotations, ]
        else:
            raise ValueError(f"Invalid annotations value: {annotations}")

        # Update surface coordinates for each annotation
        for key in annotations:
            # Get the current annotation's surface addresses
            surface_annotation = self.surface_annotations.get(key, {})
            flatmap_address = surface_annotation.get("addresses", None)

            # If no surface addresses, set surface annotation coordinates to None
            if flatmap_address is not None:
                # Calculate surface coordinates
                surface_coordinates = (
                    self._flatmap_to_surface(flatmap_address, self.coordinates))
                
                # Store surface annotation coordinates
                self.surface_annotations[key]["coordinates"] = surface_coordinates

        
    def update_surface_annotations(self, annotations = None):
        """Update cortical surface annotations based on current state."""
        # Initialize surface annotations dictionary if not present
        if annotations is None: self.surface_annotations = {} 

        # Get the list of annotations to update
        self.update_surface_addresses(annotations)
        self.update_surface_coordinates(annotations)

    # [Cortex Viewer] Observer Methods -----------------------------------------

    def observe_dataset_index(self, callback):
        """Assign a callback function to dataset value changes."""
        self.annotation_widgets.observe(callback, names = "selected_index")


    def observe_targets(self, callback):
        """Assign a callback function to target value changes."""
        for annotation_widget in self.annotation_widgets.children:
            selection_panel = annotation_widget.control_panel.selection_panel
            for target_dropdown in selection_panel.target_dropdowns.values():
                target_dropdown.observe(callback, names = "value")


    def observe_annotation(self, callback):
        """Assign a callback function to annotation data changes."""
        for annotation_widget in self.annotation_widgets.children:
            selection_panel = annotation_widget.control_panel.selection_panel
            annotations_dropdown = selection_panel.annotations_dropdown
            annotations_dropdown.observe(callback, names = "value")


    def observe_annotation_change(self, callback):
        """Assign a callback function to annotation data changes."""
        for annotation_widget in self.annotation_widgets.children:
            annotation_figure = annotation_widget.figure_panel
            annotation_figure.observe(callback, names = "_annotation_change")


    def observe_annotation_styles(self, callback):
        """Assign a callback function to style option changes."""
        for annotation_widget in self.annotation_widgets.children:
            style_panel = annotation_widget.control_panel.style_panel
            style_panel.visible_checkbox.observe(callback, names = "value")
            style_panel.color_picker.observe(callback, names = "value")


    def observe_variable_label(self, callback):
        """Assign a callback function to variable annotation label changes."""
        for annotation_widget in self.annotation_widgets.children:
            selection_panel = annotation_widget.control_panel.selection_panel
            selection_panel.variable_select.observe(callback, names = "label")

# The Cortex Viewer Widget -----------------------------------------------------

class CortexViewer(ipw.GridBox):
    
    """The Cortex Viewer widget.

    The `CortexViewer` type handles the 3D Cortex figure that renders the 
    cortical mesh that assists the flatmap viewer.
    """

    def __init__(
            self, annotation_widgets, dataset_directory, 
            control_panel_background_color = "#f0f0f0",
            panel_width = 270, panel_height = None
        ):
        # Initialize the Cortex Viewer state
        self.state = CortexViewerState(
            annotation_widgets = annotation_widgets,
            dataset_directory  = dataset_directory,
        )

        # Create the Cortex Viewer control panel
        self.control_panel = CortexControlPanel(
            self.state, control_panel_background_color
        )

        # Create the Cortex Viewer figure panel
        self.figure_panel = CortexFigurePanel(
            self.state, height = panel_height
        )

        # Initialize the GridBox with the control panel and figure panel
        children = [
            # self._make_html_header(),
            self.control_panel,
            self.figure_panel
        ]
        super().__init__(
            children = children, 
            layout   = ipw.Layout(
                border  = "1px solid rgb(159, 159, 159)", 
                padding = "15px",
                grid_template_columns = f"{panel_width}px 1fr",
                grid_template_rows    = "auto"
            )
        )
        self.add_class("cortex-viewer")

        # Assign information box observers
        for key in self._infobox_observers.keys():
            self._infobox_observers[key](partial(self.on_selection_change, key))

        # Assign user annotation input observers
        self.state.observe_annotation_change(self.on_annotation_change)

        # Assign variable annotation label observers
        self.state.observe_variable_label(self.on_variable_label_change)

        # Assign style option observers
        for key in self._style_observers.keys():
            self._style_observers[key](partial(self.on_style_change, key))

        # Assign view and drawing observers
        self.control_panel.observe_camera_dropdown(self.on_camera_mode_change)
        self.control_panel.observe_height_slider(self.on_height_change)
        self.control_panel.observe_draw_checkbox(self.on_draw_toggle)
        self.control_panel.reset_button.on_click(
            lambda _: self.figure_panel.reset_camera())
        self.control_panel.delete_button.on_click(self.on_delete_point)
        self.control_panel.observe_click_mode_toggle(self.on_click_mode_change)
        self.control_panel.clear_crosshair_button.on_click(
            lambda _: self.clear_crosshair())

        # Clicks on the mesh add points to the active annotation (or place the
        # crosshair), and crosshair clicks on any flatmap are passed here.
        self.figure_panel.mesh_click_handler = self.on_mesh_click
        for flat_panel in self._all_figure_panels():
            flat_panel.crosshair_handler = self.on_flatmap_crosshair

    # [Cortex Viewer] View and Drawing Handlers --------------------------------

    def on_camera_mode_change(self, change):
        """Handle changes to the camera (rotation) mode."""
        self.state.view["camera_mode"] = change.new
        self.figure_panel.set_camera_mode(change.new)


    def on_height_change(self, change):
        """Handle changes to the 3D panel height."""
        self.state.view["height"] = change.new
        self.figure_panel.set_height(change.new)


    def on_draw_toggle(self, change):
        """Turn drawing on the mesh on or off."""
        self.state.view["draw_on_mesh"] = bool(change.new)
        self._update_mesh_clicks()


    def _update_mesh_clicks(self):
        """Listen for mesh clicks when drawing on the mesh or in crosshair mode."""
        view = self.state.view
        self.figure_panel.set_draw_mode(
            view["draw_on_mesh"] or view["click_mode"] == "crosshair")

    # [Cortex Viewer] Crosshair Handlers ---------------------------------------

    def _all_figure_panels(self):
        """The flatmap figure panels of all datasets."""
        return [ tool.figure_panel
                 for tool in self.state.annotation_widgets.children ]


    def on_click_mode_change(self, change):
        """Switch between adding points and placing a crosshair."""
        self.state.view["click_mode"] = change.new
        crosshair = (change.new == "crosshair")
        for flat_panel in self._all_figure_panels():
            flat_panel.crosshair_mode = crosshair
        self._update_mesh_clicks()


    def clear_crosshair(self):
        """Remove the crosshair from all maps and the mesh."""
        self.state.crosshair = None
        for flat_panel in self._all_figure_panels():
            if flat_panel.crosshair is not None:
                flat_panel.set_crosshair(None)
        self.figure_panel.refresh_crosshair()


    def on_flatmap_crosshair(self, point):
        """A crosshair was placed on a flatmap: show the same spot on the mesh."""
        self.state.set_crosshair_from_flatmap(point)
        self.figure_panel.refresh_crosshair()


    def _place_crosshair_from_mesh(self, content):
        """A crosshair click on the mesh: show the spot on the flatmaps too."""
        flat_panel = self.state._get_active_figure_panel()
        point = self.state.set_crosshair_from_surface(
            content.get("position"), content.get("face"))
        flat_panel.set_crosshair(point)
        if point is None:
            self._flash_message(flat_panel,
                "That spot is outside the flatmap region.")
        self.figure_panel.refresh_crosshair()


    def _flash_message(self, figure_panel, message, seconds = 2.5):
        """Show a short message on the flatmap canvas, then clear it."""
        import threading
        figure_panel.write_message(message, fontsize = 24)
        threading.Timer(seconds, figure_panel.clear_message).start()


    def on_delete_point(self, _ = None):
        """Delete the point at the cursor of the active annotation."""
        figure_panel = self.state._get_active_figure_panel()
        if figure_panel.state.locked: return
        figure_panel.pop_point()


    def on_mesh_click(self, content):
        """Add a point to the active annotation where the mesh was clicked.

        In crosshair mode, the click places the crosshair instead.

        The clicked 3D point is converted to flatmap coordinates, and then
        added with the flatmap tool's own `push_point`, so it behaves exactly
        like a click on the flatmap (cursor, fixed points, dependencies, and
        saving all work the same way). The 3D view then updates through the
        usual annotation-change observer.
        """
        if self.state.view["click_mode"] == "crosshair":
            self._place_crosshair_from_mesh(content)
            return

        figure_panel = self.state._get_active_figure_panel()
        if figure_panel.state.locked: return
        if self.state.active is None:
            self._flash_message(figure_panel,
                "Select an annotation before drawing on the mesh.")
            return

        point = self.state.surface_to_flatmap(
            content.get("position"), content.get("face"))
        if point is None:
            self._flash_message(figure_panel,
                "That spot is outside the flatmap region.")
            return

        figure_panel.push_point(np.asarray(point, dtype = float))


    @property
    def _infobox_observers(self):
        """Return a list of observer functions for the Cortex Viewer state."""
        return {
            "dataset"    : self.state.observe_dataset_index,
            "targets"    : self.state.observe_targets,
            "annotation" : self.state.observe_annotation,
        } 


    @property
    def _style_observers(self):
        """Return a list of observer functions for the Cortex Viewer style."""
        return {
            "inflation_percent" : self.control_panel.observe_inflation_slider, 
            "overlay"           : self.control_panel.observe_overlay_dropdown, 
            "overlay_alpha"     : self.control_panel.observe_overlay_slider, 
            "point_size"        : self.control_panel.observe_point_size_slider, 
            "line_width"        : self.control_panel.observe_line_width_slider,
            "line_interp"       : self.control_panel.observe_line_interp_slider, 
            "annotation_style"  : self.state.observe_annotation_styles,
        }
    

    def on_selection_change(self, key, change):
        """Handle changes to the dataset selection."""
        # Update the control panel information
        if key == "dataset":
            self.state.dataset_index = change.new
            self.state.dataset       = self.state.get_dataset()
            self.state.targets       = self.state.get_targets()
            self.state.annotation    = self.state.get_annotation()
        elif key == "targets":
            self.state.targets       = self.state.get_targets()
            self.state.annotation    = self.state.get_annotation()
        else: # key == "annotation":
            self.state.annotation    = self.state.get_annotation()

        # Update the infobox displays
        for k in self.control_panel.infobox.keys():
            self.control_panel.refresh_infobox(k)

        # Update style options based on new dataset
        if key == "dataset":
            self.state.update_annot_cfg()
            self.state.update_styler()

        # A crosshair belongs to one participant and hemisphere.
        if key in ( "dataset", "targets" ):
            self.state.crosshair = None
            for flat_panel in self._all_figure_panels():
                if flat_panel.crosshair is not None:
                    flat_panel.set_crosshair(None)

        # Update the cortex viewer state based on selection change
        if key in ( "dataset", "targets" ):
            self.state.update_participant()
            self.state.update_coordinates()
            self.state.update_mesh()
            self.state.update_overlay()
            self.state.update_flatmap_annotations()
            self.state.update_surface_annotations()
            clear, cortex, points = True, True, True
        else: # key == "annotation"
            clear, cortex, points = False, False, True

        # Refresh the figure.
        self.figure_panel.refresh_figure(
            clear = clear, cortex = cortex, points = points)


    def on_annotation_change(self, _):
        """Handle update when the user changes the annotation data."""
        # Update the surface annotations based on the new annotation data
        self.state.update_surface_annotations()
        
        # Refresh the figure with annotation changes
        self.figure_panel.refresh_figure(
            clear = False, cortex = False, points = True)


    def on_variable_label_change(self, _):
        """Handle changes to the label of the active variable annotation."""
        self.control_panel.refresh_infobox("label")
        

    def on_style_change(self, key, change):
        """Handle changes to the style option changes."""
        # Change style based on key value
        if key != "annotation_style":
            self.state.style[key] = change.new

        # Update the mesh color based on the new overlay
        if key == "inflation_percent":
            self.state.update_coordinates()
            self.state.update_mesh()
            self.state.update_overlay()
            self.state.update_surface_coordinates()
            clear, cortex, points = False, True, True
        elif key == "overlay":
            self.state.update_overlay()
            clear, cortex, points = False, True, False
        elif key == "overlay_alpha":
            clear, cortex, points = False, True, False
        elif key in ( "point_size", "line_width" ):
            clear, cortex, points = False, False, True
        elif key == "line_interp":
            self.state.update_surface_annotations()
            clear, cortex, points = False, False, True
        elif key == "annotation_style":
            clear, cortex, points = False, False, True
        else: 
            raise ValueError(f"Invalid style key: {key}")
            
        # Update the figure with updated state
        self.figure_panel.refresh_figure(
            clear = clear, cortex = cortex, points = points)

# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer_panels.py

"""
Core implementation code for the CortexViewer's subpanels. 

CortexViewer is implemented in terms of two main subpanels:
1) CortexControlPanel: The control panel that contains viewer information about 
    the dataset, target keys, and annotations, as well as user controls for 
    adjusting the inflation level of the cortex plot.
2) CortexFigurePanel: The figure panel that contains the 3D cortex plot.

Each panel is implemented as an ipywidget, and the CortexViewer class
assembles these two panels into a single interface.

CortexViewer is implmented in _viewer.py.
"""

# Imports ----------------------------------------------------------------------

import k3d
import html
import time
import threading
import numpy as np
import ipywidgets as ipw
from matplotlib.colors import to_rgb

# The Cortex Control Panel  ----------------------------------------------------

class CortexControlPanel(ipw.VBox):
    """Control Panel for the Cortex Viewer Tool.

    The panel that contains the information and user controls for the Cortex 
    Viewer tool.
    """

    def __init__(self, state, background_color = "#f0f0f0"):
        """Initialize the Cortex Control Panel with the given state."""
        # Store the state.
        self.state = state

        # Extract the target keys
        self.target_keys = list(state.targets.keys())

        # Create information boxes
        self.infobox = {} # initialize infobox dictionary
        infobox_keys = ( "dataset", *self.target_keys, "annotation", "label" ) 
        for key in infobox_keys: # for each key in dataset and selection
            self.infobox[key] = self._init_infobox(key)
        self.refresh_infobox("label") # hidden without an active label
        
        # Create the inflation slider widget
        self.inflation_slider = self._init_inflation_slider()

        # Create the overlay dropdown widget
        self.overlay_dropdown = self._init_overlay_dropdown()

        # Create the overlay alpha slider widget
        self.overlay_slider = self._init_overlay_alpha_slider()

        # Create point size slider (user-defined points)
        self.point_size_slider = self._init_point_size_slider()

        # Create line size slider (interpolated)
        self.line_width_slider = self._init_line_width_slider()

        # Create the number of line points slider widget
        self.line_interp_slider = self._init_line_interp_slider()

        # Create the view widgets (camera mode, panel height, reset button)
        self.camera_dropdown  = self._init_camera_dropdown()
        self.height_slider    = self._init_height_slider()
        self.reset_button     = self._init_button("Reset view", "Return the camera to the starting view.")

        # Create the drawing widgets (draw on mesh toggle, delete point button)
        self.draw_checkbox    = self._init_draw_checkbox()
        self.delete_button    = self._init_button("Delete point", "Delete the point at the cursor of the active annotation (same as Backspace on the flatmap).")

        # Assemble the control panel with children widgets
        children = [
            # Cortex Viewer title
            self._make_section_title("Selection:"),
            # Dataset infoboxes
            self.infobox["dataset"],
            # Targets infoboxes
            *[ self.infobox[key] for key in self.target_keys ],
            # Annotation infobox
            self.infobox["annotation"],
            # Variable annotation label infobox
            self.infobox["label"],
            # Horizontal line
            self._make_hline(), 
            # Style Options title
            self._make_section_title("Style:"),
            # Inflation slider
            self.inflation_slider,
            # Overlay dropdown
            self.overlay_dropdown,
            # Overlay alpha slider
            self.overlay_slider, 
            # Point size slider
            self.point_size_slider, 
            # Line size slider
            self.line_width_slider,
            # Line interpolation slider
            self.line_interp_slider,
            # Horizontal line
            self._make_hline(),
            # View Options title
            self._make_section_title("View:"),
            # Camera mode dropdown
            self.camera_dropdown,
            # Panel height slider
            self.height_slider,
            # Reset view button
            self.reset_button,
            # Horizontal line
            self._make_hline(),
            # Drawing Options title
            self._make_section_title("Draw:"),
            # Draw on mesh checkbox
            self.draw_checkbox,
            # Delete point button
            self.delete_button,
        ]
        vbox = ipw.VBox(children, layout = { "width": "250px" })

        # Wrap the whole thing in an accordion so that it can be collapsed.
        accordion = ipw.Accordion((vbox, ), selected_index = 0)
        self.add_class("cortex-control-panel")

        # Initalize the VBox with the accordion.
        super().__init__(
            children = [ self._make_html_header(background_color), accordion ],  
            layout   = { "border": "0px", "height": "100%" }
        )        


    @classmethod
    def _make_html_header(cls, background_color = "#f0f0f0"):
        return ipw.HTML(f"""
            <style>
                .cortex-control-panel .jupyter-widget-Collapse-open {{
                    background-color: white;
                }}
                .cortex-control-panel .jupyter-widget-Collapse-header {{
                    background-color: white;
                    border-width: 0px;
                    padding: 0px;
                }}
                .cortex-control-panel .jupyter-widget-Collapse-contents {{
                    background-color: {background_color};
                    padding: 2px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: lightgray;
                }}
                .info-label {{
                    margin-right: 8px;
                    text-align: right;
                }}
                .info-value {{
                    background-color: white;
                    border: 1px solid rgb(158, 158, 158);
                    padding: 0px 8px 0px 8px;
                    text-align: left;
                }}
                .cortex-control-widgets {{
                    margin: 1% 3% 1% 3%;
                    width: 94%;
                }} 
                .cortex-control-widgets .widget-readout {{
                    min-width: 50px;
                }}
                .cortex-hline {{
                    border: 1px solid lightgray;
                    margin: 3%;
                    height: 0px;
                    width: 94%;
                }}
            </style>
        """)
    

    @classmethod
    def _make_section_title(cls, title):
        return ipw.HTML(f"<b style=\"margin: 0% 3% 0% 3%;\">{title}</b>")
    

    @classmethod
    def _make_hline(cls):
        return ipw.HTML("""<div class="cortex-hline"></div>""")


    def _prep_infobox_value(self, key):
        """Prepare the infobox value for display."""
        if key == "dataset":
            return self.state.dataset
        elif key == "annotation":
            return self.state.annotation
        elif key == "label":
            return self.state.active_label
        else: # key in state.targets
            return self.state.targets[key]


    def _make_infobox_value(self, value):
        """Make the infobox value for display."""
        return f"""<div class="info-value">{html.escape(str(value))}</div>"""
        

    def _init_infobox(self, key):
        """Update an information box for the given key and state."""
        value = self._prep_infobox_value(key)
        return ipw.Box([
            ipw.HTML(f"""<div class="info-label">{key.capitalize()}:</div>""",
                     layout = { "width": "40%", "margin": "0px" }),
            ipw.HTML(self._make_infobox_value(value), 
                     layout = { "width": "60%", "margin": "0px" }),
        ], layout = ipw.Layout(
            display = "flex",
            margin  = "1% 3% 1% 3%",
        ))


    def _init_inflation_slider(self):
        inflation_slider = ipw.IntSlider(
            value             = self.state.style["inflation_percent"],
            min               = 0,
            max               = 100,
            step              = 1,
            description       = "Inflation %:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = "d", 
        )
        inflation_slider.add_class("cortex-control-widgets")
        return inflation_slider


    def _init_overlay_dropdown(self):
        overlay_dropdown = ipw.Dropdown(
            options     = [
                ( "None", "curvature" ), 
                ( "Polar Angle", "angle" ), 
                ( "Eccentricity", "eccen" ), 
                ( "Variance Explained", "vexpl" ), 
                ( "Faces", "faces" ),
                ( "Bodies", "bodies" ),
                ( "Places", "places" ),
                ( "Words", "words" ),
            ],
            value       = self.state.style["overlay"],    
            description = "Overlay:",
        )
        overlay_dropdown.add_class("cortex-control-widgets")
        return overlay_dropdown


    def _init_overlay_alpha_slider(self):
        overlay_slider = ipw.FloatSlider(
            value             = self.state.style["overlay_alpha"],
            min               = 0.0,
            max               = 1.0,
            step              = 0.1,
            description       = "Alpha:",
            continuous_update = False,
            orientation       = "horizontal",
        )
        return overlay_slider
    

    def _init_point_size_slider(self):
        point_size_slider = ipw.FloatSlider(
            value             = self.state.style["point_size"],
            min               = 0.5,
            max               = 5,
            step              = 0.1,
            description       = "Point Size:",
            continuous_update = False,
            orientation       = "horizontal",
        )
        return point_size_slider
    

    def _init_line_width_slider(self):
        line_width_slider = ipw.FloatSlider(
            value             = self.state.style["line_width"],
            min               = 0.10,
            max               = 0.50,
            step              = 0.05,
            description       = "Line Width:",
            continuous_update = False,
            orientation       = "horizontal",
        )
        return line_width_slider


    def _init_line_interp_slider(self):
        line_interp_slider = ipw.IntSlider(
            value             = self.state.style["line_interp"],
            min               = 5,
            max               = 20,
            step              = 1,
            description       = "Line Interp.:",
            continuous_update = False,
            orientation       = "horizontal",
        )
        return line_interp_slider


    def _init_camera_dropdown(self):
        camera_dropdown = ipw.Dropdown(
            options     = [
                ( "Orbit (steady)", "orbit" ),
                ( "Trackball (free)", "trackball" ),
            ],
            value       = self.state.view["camera_mode"],
            description = "Rotation:",
        )
        camera_dropdown.add_class("cortex-control-widgets")
        return camera_dropdown


    def _init_height_slider(self):
        height_slider = ipw.IntSlider(
            value             = self.state.view["height"],
            min               = 300,
            max               = 1400,
            step              = 50,
            description       = "Height (px):",
            continuous_update = False,
            orientation       = "horizontal",
        )
        height_slider.add_class("cortex-control-widgets")
        return height_slider


    def _init_draw_checkbox(self):
        draw_checkbox = ipw.Checkbox(
            value       = self.state.view["draw_on_mesh"],
            description = "Click on mesh to add points",
            indent      = False,
            tooltip     = "When checked, clicking on the 3D mesh adds a point "
                          "to the active annotation, just like clicking on "
                          "the flatmap.",
        )
        draw_checkbox.add_class("cortex-control-widgets")
        return draw_checkbox


    def _init_button(self, description, tooltip = ""):
        button = ipw.Button(description = description, tooltip = tooltip)
        button.add_class("cortex-control-widgets")
        return button


    def refresh_infobox(self, key):
        """Refresh the control panel display to reflect updated infobox values."""
        value = self._prep_infobox_value(key)
        self.infobox[key].children[1].value = self._make_infobox_value(value)

        # An infobox without a value (no active variable annotation) is hidden.
        self.infobox[key].layout.display = "none" if value is None else "flex"


    def observe_inflation_slider(self, callback): 
        """A method to register a callback for changes in the inflation slider."""
        self.inflation_slider.observe(callback, names = "value")


    def observe_overlay_dropdown(self, callback):
        """A method to register a callback for changes in the overlay dropdown."""
        self.overlay_dropdown.observe(callback, names = "value")

    
    def observe_overlay_slider(self, callback):
        """A method to register a callback for changes in the overlay alpha slider."""
        self.overlay_slider.observe(callback, names = "value")


    def observe_point_size_slider(self, callback):
        """A method to register a callback for changes in the point size slider."""
        self.point_size_slider.observe(callback, names = "value")


    def observe_line_width_slider(self, callback):
        """A method to register a callback for changes in the line width slider."""
        self.line_width_slider.observe(callback, names = "value")

    
    def observe_line_interp_slider(self, callback):
        """A method to register a callback for changes in the line points slider."""
        self.line_interp_slider.observe(callback, names = "value")


    def observe_camera_dropdown(self, callback):
        """A method to register a callback for changes in the camera mode."""
        self.camera_dropdown.observe(callback, names = "value")


    def observe_height_slider(self, callback):
        """A method to register a callback for changes in the panel height."""
        self.height_slider.observe(callback, names = "value")


    def observe_draw_checkbox(self, callback):
        """A method to register a callback for the draw on mesh checkbox."""
        self.draw_checkbox.observe(callback, names = "value")

# The Cortex Figure Panel ------------------------------------------------------

class CortexFigurePanel(ipw.GridBox):
    """Cortex Figure Panel.

    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    # Camera field of view (degrees). A narrower field of view with the camera
    # farther away gives less perspective distortion and steadier rotation
    # than the original 60 degrees.
    CAMERA_FOV = 35

    # Starting camera: [position (3), target (3), up vector (3)]. Same viewing
    # direction as the original [-160, -10, -6, 15, -30, 0, 0, 0, 1], with the
    # camera moved back so the brain fills a similar part of the panel at the
    # narrower field of view.
    DEFAULT_CAMERA = [-305, 7, -11, 15, -30, 0, 0, 0, 1]

    # Clicks closer together in time than this (seconds) at the same spot are
    # treated as one click (the cortex and overlay meshes overlap exactly, so
    # one click can be reported by both).
    CLICK_DEBOUNCE = 0.3

    def __init__(self, state, width = None, height = None):
        # Store the state.
        self.state = state
        if height is None: height = self.state.view["height"]

        # Handler for clicks on the mesh, set by the CortexViewer. It is called
        # with the k3d click message (a dict with "position" and "face").
        self.mesh_click_handler = None
        self._last_click = (None, 0.0)

        # Create a figure background (k3d plot)
        plot_kwargs = dict(
            # K3d does not use width, just fills the space
            height            = height,
            grid_visible      = False,
            camera_auto_fit   = False,
            menu_visibility   = False,
            camera_fov        = self.CAMERA_FOV,
            axes_helper       = 0, # remove axes direction helper
            camera_zoom_speed = 1.5,
            camera_mode       = self.state.view["camera_mode"],
        )
        try:
            self.figure = k3d.plot(**plot_kwargs)
        except TypeError: # older k3d without camera_mode
            plot_kwargs.pop("camera_mode")
            self.figure = k3d.plot(**plot_kwargs)

        # Create and add the cortex mesh to the figure
        self.k3dmesh_cortex = self._init_cortex()

        # Create and add the overlay mesh to the figure
        self.k3dmesh_overlay = self._init_overlay()
        
        # Create active annotation to figure 
        self.k3dline_active, self.k3dpoints_active = \
            self._init_active_annotation()

        # Create background annotations to figure
        self.k3dline_background, self.k3dpoints_background = \
            self._init_background_annotations()

        # Add the meshes and points to the figure and initial render
        self.figure += self.k3dmesh_cortex
        self.figure += self.k3dmesh_overlay 
        self.figure += self.k3dline_active
        self.figure += self.k3dpoints_active 
        self.figure += self.k3dline_background
        self.figure += self.k3dpoints_background 

        # Set initial camera values
        self.reset_camera()

        # Listen for clicks on the cortex and overlay meshes. Clicks are only
        # reported by k3d when the plot is in "callback" mode (see
        # set_draw_mode), so this has no effect until drawing is turned on.
        self.k3dmesh_cortex.click_callback  = self._on_k3d_click
        self.k3dmesh_overlay.click_callback = self._on_k3d_click
        self.set_draw_mode(self.state.view["draw_on_mesh"])

        # Define layout (GridBox as HBox)
        layout = ipw.Layout(
            grid_template_columns = "1fr",
            height = f"{height}px",
            width  = f"{width}px" if width is not None else "100%"
        )
        super().__init__([ self.figure ], layout = layout)
        self.figure.camera_auto_fit = False

    # View Methods -------------------------------------------------------------

    def reset_camera(self):
        """Move the camera back to the starting view."""
        self.figure.camera = list(self.DEFAULT_CAMERA)
        # Send even if the stored value did not change, so the browser always
        # gets it.
        self.figure.send_state("camera")


    def set_camera_mode(self, mode):
        """Set the k3d camera mode ("orbit" or "trackball")."""
        if not hasattr(self.figure, "camera_mode"): return
        if mode == self.figure.camera_mode: return

        # When the mode changes, k3d (in the browser) re-fits the camera to
        # the whole scene and reports the new camera back to Python. That
        # moves the view. To keep the current view, wait for that report and
        # then send the old camera again. (Sending it right away does not
        # work: the browser's report arrives later and wins.)
        camera = list(self.figure.camera)

        def restore(change):
            self._stop_watching_camera(restore)
            if list(change.new) != camera:
                self.figure.camera = camera

        self.figure.observe(restore, names = "camera")
        self.figure.camera_mode = mode
        # Stop waiting if the report never comes (e.g. the view is hidden).
        threading.Timer(2.0, self._stop_watching_camera, [restore]).start()


    def _stop_watching_camera(self, handler):
        """Remove a camera observer, if it is still registered."""
        try:
            self.figure.unobserve(handler, names = "camera")
        except ValueError:
            pass


    def set_height(self, height):
        """Set the height (in pixels) of the 3D panel."""
        height = int(height)
        # k3d uses `height` only when a plot is first drawn (new views).
        self.figure.height = height
        # Plots already on screen do not follow `height` (k3d 3.1), so resize
        # them with a short script. k3d runs `additional_js_code` in every
        # view of the plot when it changes.
        self.figure.additional_js_code = (
            "(function(){"
            "var n = K3DInstance.getWorld().targetDOMNode.parentElement;"
            f"n.style.height = '{height}px';"
            "K3DInstance.resizeHelper();"
            "})();"
        )
        self.layout.height = f"{height}px"


    def set_draw_mode(self, draw):
        """Turn drawing on the mesh on (k3d "callback" mode) or off ("view")."""
        self.figure.mode = "callback" if draw else "view"

    # Click Handling Methods ---------------------------------------------------

    def _on_k3d_click(self, content):
        """Receive a click on the mesh from k3d and pass it to the handler."""
        if self.mesh_click_handler is None: return
        if not isinstance(content, dict) or "position" not in content: return

        # Ignore the second report of the same click (from the other mesh).
        now = time.monotonic()
        position = tuple(np.round(np.asarray(content["position"], float), 4))
        last_position, last_time = self._last_click
        if position == last_position and now - last_time < self.CLICK_DEBOUNCE:
            return
        self._last_click = (position, now)

        self.mesh_click_handler(content)


    # k3d Color Helper Method --------------------------------------------------

    def _rgb_to_k3dcolor(self, colors):
        """Converts a matplotlib color (RGB) into a hex integer for k3d.

        If the given color is a matrix of RGB triples, then a list of
        integers, one per row, is returned. 
        """
        # Convert to numpy array for easier processing
        colors = np.array(colors)
        
        # Handle string color inputs (e.g. "red", "#ff0000", etc.)
        if np.issubdtype(colors.dtype, np.str_):
            if colors.ndim == 0: colors = colors.reshape(-1,)
            colors = np.array([to_rgb(x) for x in colors], dtype = float)

        # Handle single RGB or RGBA triple input (e.g. [1, 0, 0] or [1, 0, 0, 1])
        if colors.ndim == 1: colors = colors.reshape((1, -1))
            
        # Handle floating point inputs by converting to uint8
        if np.issubdtype(colors.dtype, np.floating):
            if colors.max() > 1.0: # if max is greater than 1, assume 0-255 range
                colors = colors.astype(np.uint8)
            else: # else assume 0-1 range and convert to 0-255
                colors = (colors * 255).astype(np.uint8)

        # Handle integer inputs by checking if they are within uint8 range and converting to uint8
        if np.issubdtype(colors.dtype, np.integer):
            if colors.min() < 0 or colors.max() > 255:
                raise ValueError("Color values must be within uint8 range [0-255].")    
            colors = colors.astype(np.uint8)

        # Check that colors are now uint8 and converted to 2D array
        if not np.issubdtype(colors.dtype, np.uint8):
            raise ValueError("Color values must be convertible from float [0,1] or uint8 [0,255].")
        if colors.ndim != 2:
            raise ValueError("Color input must be scalar, 1D RGB/RGBA, or 2D Nx3/Nx4.")
        
        # Convert RGB/RGBA values to k3d color integers
        colors = colors.astype(np.uint32) # ensure uint32 for bitwise operations
        if colors.shape[1] == 3: # if RGB, convert to k3d color integer
            return np.array(
                [ ((r << 16) | (g << 8) | b) for r, g, b in colors ], 
                dtype = np.uint32
            )
        elif colors.shape[1] == 4: # if RGBA, ignore the alpha channel
            # NOTE: k3d does not support alpha in the color integer, ignor
            return np.array(
                [ ((r << 16) | (g << 8) | b) for r, g, b, _ in colors ], 
                dtype = np.uint32
            )
        else:
            raise ValueError("Color matrices must be RGB (Nx3) or RGBA (Nx4).")

    # Empty Value Methods ------------------------------------------------------

    def _empty_coordinates(self):
        """Helper method to create an empty matrix for initializing empty plots."""
        return np.array([[0, 0, 0]], dtype = np.float32)


    def _empty_indices(self):
        """Helper method to create an empty matrix for initializing empty meshes."""
        return np.array([[0, 0, 0]], dtype = np.uint32)


    def _empty_colors(self):
        """Helper method to create an empty color for initializing empty plots."""
        return np.array([0x000000], dtype = np.uint32)
    
    # Initialize Methods -------------------------------------------------------

    def _init_mesh(self):
        """Initialize an empty and invisible mesh."""
        mesh = k3d.mesh(
            vertices     = self._empty_coordinates(), 
            indices      = self._empty_indices(),
            colors       = self._empty_colors(),
            wireframe    = False,
            flat_shading = False
        )
        mesh.visible = False
        return mesh


    def _init_cortex(self):
        """Initialize the cortex mesh."""
        cortex_kwargs = self._prep_cortex()
        return k3d.mesh(**cortex_kwargs, wireframe = False, flat_shading = False)          
    

    def _init_overlay(self):
        """Initialize the cortex overlay mesh."""
        overlay_kwargs = self._prep_overlay()
        if overlay_kwargs is None:
            return self._init_mesh()
        return k3d.mesh(**overlay_kwargs, wireframe = False, flat_shading = False)


    def _init_points(self):
        """Initialize an empty and invisible points plot."""
        points = k3d.points(
            positions = self._empty_coordinates(),
            colors    = self._empty_colors(), 
            shader    = "3d"
        )
        points.visible = False
        return points


    def _init_line(self):
        """Initialize an empty and invisible line plot."""
        line = k3d.line(
            vertices = self._empty_coordinates(),
            colors   = self._empty_colors(), 
            width    = 0.1, 
            shader   = "mesh"
        )
        line.visible = False
        return line


    def _init_active_annotation(self):
        """Initialize the line and points for the active annotation."""
        active_kwargs = self._prep_active_annotation()
        if active_kwargs is None:
            return ( self._init_line(), self._init_points() )
        return (
            k3d.line(**active_kwargs["line"], shader = "mesh"),
            k3d.points(**active_kwargs["points"], shader = "3d"), 
        )


    def _init_background_annotations(self):
        """Initialize the points plot for the background annotations."""
        background_kwargs = self._prep_background_annotations()
        if background_kwargs is None:
            return ( self._init_line(), self._init_points() )
        return (
            k3d.line(**background_kwargs["line"], shader = "mesh"),
            k3d.points(**background_kwargs["points"], shader = "3d")
        )

    # Prepare Cortex Methods ---------------------------------------------------

    def _prep_cortex(self):
        """Prepare the data for the cortex mesh."""
        vertices  = self.state.coordinates.T # (n_vertices, 3)
        indices   = self.state.fsaverage[self.state.hemisphere]["tesselation"]
        curvature = self._rgb_to_k3dcolor(self.state.curvature)
        return { 
            "vertices" : vertices.astype(np.float32), 
            "indices"  : indices.T.astype(np.uint32), 
            "colors"   : curvature.astype(np.uint32) 
        }
    
    # Prepare Overlay Methods --------------------------------------------------

    def _prep_overlay(self):
        """Prepare the data for the cortex overlay mesh."""
        # If no overlay, return None
        if self.state.style["overlay"] == "curvature":
            return None

        # Else return the overlay mesh keyword arguments
        return {
            **self._prep_cortex(), # same vertices + indices
            "colors"   : self._rgb_to_k3dcolor(self.state.overlay),
            "opacity"  : float(self.state.style["overlay_alpha"])
        }
    
    # Prepare Active Points Methods ---------------------------------------------------

    def _prep_active_annotation(self):
        """Prepare the data for the active annotation."""
        # Get the current active surface annotation (if there is one)
        active             = self.state.active
        surface_annotation = self.state.surface_annotations.get(active, {})

        # If no coordinates, return None to skip plotting.
        coordinates = surface_annotation.get("coordinates", None)
        if coordinates is None or coordinates.shape[1] == 0: return None

        # Get the annotation style from the styler (active = None)
        # If not visible, return None to skip plotting.
        annotation_style = self.state.styler(None)
        if not annotation_style["visible"]: return None

        # Get number of annotation vertex (line) coordinates and point types
        vertices    = coordinates.T.astype(np.float32)
        positions   = vertices.copy() # copy!
        point_types = surface_annotation.get("point_types", None)

        # Check if vertices are all fixed points, skip lines if so. 
        if np.all(point_types == self.state.POINT_FIXED):
            vertices = self._empty_coordinates() # set vertices to empty to skip line plotting

        # Get annotation positions (for points) and point types
        positions   = positions[point_types != self.state.POINT_INTERP]
        point_types = point_types[point_types != self.state.POINT_INTERP]
        n_points    = positions.shape[0]

        # Prepare scatter sizes by points type (slightly larger fixed points)
        point_sizes = np.full(n_points, self.state.style["point_size"])
        point_sizes[point_types == self.state.POINT_FIXED] = self.state.style["point_size"] * 1.25

        # Prepare colors for each annotation point
        annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])

        # Return the active annotation plot keyword arguments by plot type
        return { 
            "line": {
                "vertices" : vertices.astype(np.float32),
                "width"    : float(self.state.style["line_width"]),
                "colors"   : np.full(vertices.shape[0], annotation_color, dtype = np.uint32)
            },
            "points": {
                "positions"   : positions.astype(np.float32), 
                "point_sizes" : point_sizes.astype(np.float32), 
                "colors"      : np.full(n_points, annotation_color, dtype = np.uint32)
            }
        }

    # Prepare Background Points Methods ----------------------------------------

    def _prep_background_annotations(self):
        """Prepare the data for the background annotations."""
        # Get the list of annotations excluding the active one
        active = self.state.active
        annotation_list = [
            annotation for annotation in self.state.surface_annotations.keys()
            if annotation != active
        ]
        
        # Initialize empty arrays for all coordinates and colors
        all_vertices  = np.empty((0, 3)) 
        all_positions = np.empty((0, 3))
        all_lcolors   = np.empty((0,), dtype = np.uint32)
        all_pcolors   = np.empty((0,), dtype = np.uint32)

        # Initailize NaN array to separate annotations (for line plotting)
        coord_sep = np.full((1, 3), np.nan)
        color_sep = np.array([0], dtype = np.uint32)

        for annotation in annotation_list: # for each annotation
            # Get the surface annotation and style for the annotation
            surface_annotation = self.state.surface_annotations[annotation]

            # If no coordinates, skip processing.
            coordinates = surface_annotation.get("coordinates", None)
            if coordinates is None or coordinates.shape[1] == 0: continue

            # Get the annotation style from the styler (active = None)
            # If not visible, return None to skip plotting.
            annotation_style = self.state.styler(self.state.template(annotation))
            if not annotation_style["visible"]: continue

            # Get annotation color and point types for the current annotation
            annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])
            point_types = surface_annotation.get("point_types", None)

            # Get number of annotation vertex (line) coordinates and point types
            vertices  = coordinates.T.astype(np.float32)
            positions = vertices.copy() # copy!

            # Check if not all vertices are all fixed points, all to lines.
            if not np.all(point_types == self.state.POINT_FIXED):
                # Prepare the vertices and line colors arrays
                all_vertices  = np.vstack((all_vertices, vertices, coord_sep))
                vertex_colors = np.full(vertices.shape[0], annotation_color)
                all_lcolors   = np.hstack((all_lcolors, vertex_colors, color_sep))

            # Get annotation positions (for points) and point types
            positions   = positions[point_types != self.state.POINT_INTERP]
            point_types = point_types[point_types != self.state.POINT_INTERP]

            # Prepare the positions and point colors arrays
            all_positions = np.vstack((all_positions, positions))
            point_colors  = np.full(positions.shape[0], annotation_color)
            all_pcolors   = np.hstack((all_pcolors, point_colors))
    
        # If no coordinates, return None to skip plotting.
        if all_vertices.shape[0] == 0: return None

        return { 
            "line": {
                "vertices" : all_vertices.astype(np.float32),
                "width"    : float(self.state.style["line_width"] * 0.5), 
                "colors"   : all_lcolors.astype(np.uint32)
            },
            "points": {
                "positions"  : all_positions.astype(np.float32), 
                "point_size" : float(self.state.style["point_size"] * 0.5),
                "colors"     : all_pcolors.astype(np.uint32)
            }
        }
    
    # Figure Clear Method ------------------------------------------------------

    def clear_figure(self):
        """Clear the figure by setting layers to invisible."""
        self.k3dmesh_cortex.visible       = False
        self.k3dmesh_overlay.visible      = False
        self.k3dline_active.visible       = False
        self.k3dpoints_active.visible     = False
        self.k3dline_background.visible   = False
        self.k3dpoints_background.visible = False 

    # Figure Refresh Methods ---------------------------------------------------

    def refresh_cortex(self):
        """Refresh the cortex mesh."""
        # Update the cortex mesh values
        cortex_kwargs = self._prep_cortex()
        moved = self._vertices_changed(self.k3dmesh_cortex, cortex_kwargs)
        for key in cortex_kwargs.keys():
            setattr(self.k3dmesh_cortex, key, cortex_kwargs[key])
        if moved: self._rearm_clicks(self.k3dmesh_cortex)
        self.k3dmesh_cortex.visible = True

        # Update the overlay mesh values
        overlay_kwargs = self._prep_overlay()
        if self.state.style["overlay"] == "curvature":
            self.k3dmesh_overlay.visible = False
        else:
            moved = self._vertices_changed(self.k3dmesh_overlay, overlay_kwargs)
            for key in overlay_kwargs.keys():
                setattr(self.k3dmesh_overlay, key, overlay_kwargs[key])
            if moved: self._rearm_clicks(self.k3dmesh_overlay)
            self.k3dmesh_overlay.visible = True


    @staticmethod
    def _vertices_changed(mesh, kwargs):
        """Whether `kwargs` would move or replace the vertices of `mesh`."""
        return (
            not np.array_equal(mesh.vertices, kwargs["vertices"]) or
            not np.array_equal(mesh.indices,  kwargs["indices"])
        )


    @staticmethod
    def _rearm_clicks(mesh):
        """Make k3d rebuild its click lookup after the mesh vertices change.

        k3d finds the clicked triangle with a search tree that it builds once
        and does not rebuild when the vertices are updated in place (e.g. when
        the inflation changes). Clicks would then be tested against the old
        shape, and most would be missed. Turning the click callback off and on
        again makes k3d rebuild the tree from the current vertices.
        """
        callback = mesh.click_callback
        if callback is None: return
        mesh.click_callback = None
        mesh.click_callback = callback

    
    def refresh_points(self):
        """Refresh the annotation points."""
        # Update the surface active annotation
        active_kwargs = self._prep_active_annotation()
        if active_kwargs is None:
            self.k3dline_active.visible   = False
            self.k3dpoints_active.visible = False
        else:
            # Update the active line layer (interpolated between points)
            line_kwargs = active_kwargs["line"]
            for key in line_kwargs.keys(): 
                setattr(self.k3dline_active, key, line_kwargs[key])
            self.k3dline_active.visible = True

            # Update the active points layer (fixed + user points)
            points_kwargs = active_kwargs["points"]
            for key in points_kwargs.keys():
                setattr(self.k3dpoints_active, key, points_kwargs[key])
            self.k3dpoints_active.visible = True
    
        # Update the surface background annotation
        background_kwargs = self._prep_background_annotations()
        if background_kwargs is None:
            self.k3dline_background.visible   = False
            self.k3dpoints_background.visible = False
        else:
            # Update the background line layer (interpolated between points)
            line_kwargs = background_kwargs["line"]
            for key in line_kwargs.keys():
                setattr(self.k3dline_background, key, line_kwargs[key])
            self.k3dline_background.visible = True

            # Update the background points layer (fixed + user points)
            points_kwargs = background_kwargs["points"]
            for key in points_kwargs.keys():
                setattr(self.k3dpoints_background, key, points_kwargs[key])
            self.k3dpoints_background.visible = True
            

    def refresh_figure(self, clear = False, cortex = True, points = True):
        """Refresh the entire figure."""
        # Disable auto rendering for performance
        self.figure.auto_rendering = False

        # Figure refresh, dependent on which layers need to be updated
        if clear:
            self.clear_figure()
        if cortex:
            self.refresh_cortex()
        if points:
            self.refresh_points()
        
        # Re-enable auto rendering after cortex and overlay values
        self.figure.auto_rendering = True
        self.figure.render()

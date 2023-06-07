r"""quad view"""

from .base import Base
from vizer import utils
from vizer.readers import RawConfig
import os.path
import re
import numpy

from paraview import simple, vtk
from trame.widgets import vuetify, paraview, html
from trame.app import get_server, asynchronous

log = utils.get_logger(__name__)

class CONSTANTS:
    Colors = [ [1., 0., 0.], [1., 1., 0.],  [0., 1., 0.] ]
    AxisNames = ['x', 'y', 'z']
    OutlinePropertyNames = ['XSlices', 'YSlices', 'ZSlices']

from vtkmodules.vtkRenderingCore import vtkTextActor

class ScaleActor(vtkTextActor):
    def __init__(self, config: RawConfig):
        super().__init__()
        self._config = config
        self._setup()

    def update_scale(self, scale):
        if self._config:
            q = self._config.spacing
            self.SetInput(f'scale: 1 px = {q.scale(scale):#.2q}')
            self.SetVisibility(True)
        else:
            # nothing to do since we don't have any scale information
            self.SetInput(' ')
            self.SetVisibility(False)

    def _setup(self):
        self.SetInput('[missing update]')
        self.SetPosition(20, 10)
        self.SetTextScaleModeToNone()
        self.GetTextProperty().SetFontSize(16)
        self.GetTextProperty().SetColor(0, 0, 0)
        self.GetTextProperty().SetFontFamilyToArial()
        self.GetTextProperty().FrameOn()
        self.GetTextProperty().SetFrameWidth(3)
        self.GetTextProperty().SetBackgroundRGBA(1, 1, 1, 1)

class UIBuilder:

    def toggle_callback(self, view, var, value):
        view._state[var] = value
        view.update_client_state()

    def toggle_button(self, view, var, on_icon, off_icon, on_text, off_text, **kwargs):
        if 'click' in kwargs:
            callback = kwargs['click']
            click_callback = lambda _: callback()
        else:
            click_callback = lambda value: self.toggle_callback(view, var, value)

        with vuetify.VTooltip(left=True, v_if=f'{view.id}_{var}'):
            with vuetify.Template(v_slot_activator="{on, attrs}"):
                vuetify.VIcon(on_icon,
                    click=lambda **_: click_callback(False),
                    classes="mr-4",
                    v_bind="attrs",
                    v_on="on",
                    __properties=[("v_bind", "v-bind"), ("v_on", "v-on")])
            html.Pre(on_text)
        with vuetify.VTooltip(left=True, v_if=f'!{view.id}_{var}'):
            with vuetify.Template(v_slot_activator="{on, attrs}"):
                vuetify.VIcon(off_icon,
                    click=lambda **_: click_callback(True),
                    classes="mr-4",
                    v_bind="attrs",
                    v_on="on",
                    __properties=[("v_bind", "v-bind"), ("v_on", "v-on")])
            html.Pre(off_text)

    def maximize_button(self, view, i, j):
        with vuetify.VTooltip(left=True, v_if=f'{view.id}_no_maximized'):
            with vuetify.Template(v_slot_activator="{on, attrs}"):
                vuetify.VIcon("mdi-border-all",
                    click=lambda **_: view.toggle_maximize(i, j),
                    classes="mr-4",
                    v_bind="attrs",
                    v_on="on",
                    __properties=[("v_bind", "v-bind"), ("v_on", "v-on")])
            html.Pre("Maximize")
        with vuetify.VTooltip(left=True, v_if=f'!{view.id}_no_maximized'):
            with vuetify.Template(v_slot_activator="{on, attrs}"):
                vuetify.VIcon("mdi-window-maximize",
                    click=lambda **_: view.toggle_maximize(i, j),
                    classes="mr-4",
                    v_bind="attrs",
                    v_on="on",
                    __properties=[("v_bind", "v-bind"), ("v_on", "v-on")])
            html.Pre("Restore")

    def slice_slider(self, view, axis):
        vuetify.VSlider(dense=True, hide_details=True,
            min=(f'{view.id}_slice_min_{axis}', 0), max=(f'{view.id}_slice_max_{axis}', 0),
            v_model=(f'{view.id}_slice_{axis}', 0))

class Quad(Base):
    """A quad view that renders the dataset in four views."""
    def __init__(self, filename, opts, **kwargs) -> None:
        super().__init__(filename, opts)
        self._ui_builder = kwargs.get('ui_builder', UIBuilder())
        self._views = [None, None, None, None]
        self._html_views = [None, None, None, None]
        self._outline = None
        self._slices = [None, None, None]
        self._outer_slices = [None] * 6
        self._active_subsampling_factor = self.subsampling_factor
        self._force_outer_slices = kwargs.get('force_outer_slices', False)
        self._use_vlayout = kwargs.get('use_vlayout', False)

        # storing only data dependent state variables
        # that need to be updated on the client as well
        self._state = {}
        for axis in range(3):
            self._state[f'slice_min_{axis}'] = 0
            self._state[f'slice_max_{axis}'] = 0
            self._state[f'slice_{axis}'] = -1

        # next, state we want linked between views when requested.
        # self._state['full_res'] = self._full_res
        self._state['show_inner_slices'] = False if self._force_outer_slices else True
        self._state['full_res'] = False
        self._state['max_row'] = 0
        self._state['max_col'] = 0
        self._state['no_maximized'] = True
        self._block_update = False

        # load the color categories, if present
        self.load_categories()

    @property
    def state(self):
        return self._state

    def get_scalar_name(self):
        """returns the scalar array name"""
        scalars = self.producer.GetDataInformation().GetPointDataInformation().GetAttributeInformation(
            vtk.vtkDataSetAttributes.SCALARS)
        return scalars.GetName()

    def update_client_state(self):
        """updates the client with the current state."""
        state = get_server().state
        new_state = {}
        for x, y in self._state.items():
            new_state[f'{self.id}_{x}'] = y
        state.update(new_state)

    def copy_state_from(self, other):
        """copies the state from the other view."""
        if self._state['full_res'] != other._state['full_res']:
            self.toggle_full_res()
        elif not self._block_update and self._state != other._state:
            # log.info(f'propagating changes from {other.id} to {self.id} ...')
            self._state.update(other._state)
            self.update_client_state()

    def _link_interaction(self):
        for other in Base.get_linked_views(self):
            for axis in range(4):
                other._views[axis].CameraPosition = self._views[axis].CameraPosition
                other._views[axis].CameraFocalPoint = self._views[axis].CameraFocalPoint
                other._views[axis].CameraViewUp = self._views[axis].CameraViewUp
                other._views[axis].CameraParallelScale = self._views[axis].CameraParallelScale
                other._html_views[axis].update()

    @property
    def annotations_txt(self):
        """returns the annotations for this view."""
        annotations = self._annotations[:]
        annotations.append(f'subsampling: {self._active_subsampling_factor}X')
        return '\n'.join(annotations)

    @staticmethod
    def can_show(meta):
        """returns true if this view can show the dataset"""
        return meta.is_structured and not meta.is_empty()

    def subsample(self, dataset):
        """overridden to support full resolution mode."""
        return super().subsample(dataset) if not self._state['full_res'] else dataset

    def toggle_full_res(self):
        """toggles the full resolution mode."""
        self._state['full_res'] = not self._state['full_res']

        # update the slice positions to match the subsampling factor being used
        for axis in range(3):
            val = self._state[f'slice_{axis}']
            self._state[f'slice_{axis}'] = \
                val * self.subsampling_factor if self._state['full_res'] else val // self.subsampling_factor
        self._active_subsampling_factor = self.subsampling_factor if not self._state['full_res'] else 1
        log.info(f'{self.id}: toggling full res: {self._state["full_res"]}')
        if not self._state['full_res']:
            self.set_dataset(self.producer.GetClientSideObject().GetOutputDataObject(0))
        else:
            self._block_update = True
            asynchronous.create_task(self.load_full_res())

    async def load_full_res(self):
        """loads the full resolution dataset."""
        await self.load_dataset(async_only=True)
        self._block_update = False

    def load_categories(self):
        """Loads the color categories from meta data file associated with the dataset."""
        self._categories = {}
        self._annotations = self.meta.raw_config.annotations if self.meta.raw_config is not None else []
        meta_filename = f'{os.path.splitext(self.meta.filename)[0]}.txt'
        if not os.path.exists(meta_filename):
            return

        # this is for old text files; this can be removed once we've confirmed that
        # json files are the only ones we need to support for metadata
        with open(meta_filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                lline = line.lower()
                if lline.startswith('segorder:'):
                    lline = lline[len('segorder:'):]
                    regex = r"(?:\s*(?P<value>\d+)-(?P<text>[^,]+),?)"
                    matches = re.finditer(regex, lline)
                    self._categories = dict([(int(m.group('value')), m.group('text')) for m in matches])
                elif lline.startswith('samplename:'):
                    line = 'sample name: ' + line[len('samplename:'):].strip()
                    self._annotations.append(line)
                elif lline.startswith('segmented'):
                    line = 'segmented: ' + line[len('segmented'):].strip()
                    self._annotations.append(line)
        log.info(f'{self.id}: Loaded categories: {self._categories}')

    def _copy_slice_camera(self, view):
        """Links the interaction of the given axis to the other views."""
        fp = view.CameraFocalPoint
        for i in range(3):
            target_view = self._views[i]
            if target_view == view:
                continue

            target_view.CameraParallelScale = view.CameraParallelScale

            pos = [0, 0, 0]
            for cc in range(3):
                pos[cc] = fp[cc] + target_view.CameraPosition[cc] - target_view.CameraFocalPoint[cc]

            target_view.CameraFocalPoint = fp
            target_view.CameraPosition = pos
            self._html_views[i].update()

    def toggle_maximize(self, i, j):
        if self._state['no_maximized']:
            self._state['no_maximized'] = False
            self._state['max_row'] = i
            self._state['max_col'] = j
        else:
            self._state['no_maximized'] = True
            self._state['max_row'] = -1
            self._state['max_col'] = -1
        self.update_client_state()
        Base.propagate_changes_to_linked_views(self)

    def create_slice_view(self, axis:int):
        """Creates a slice view for the given axis."""
        view = self.create_render_view()
        # setup camera position
        pos = [ [10, 0, 0], [0, -10, 0], [0, 0, 10] ]
        up = [ [0, 0, 1], [0, 0, 1], [0, 1, 0] ]
        view.CameraPosition = pos[axis]
        view.CameraViewUp = up[axis]
        view.InteractionMode = '2D'
        view.OrientationAxesVisibility = False

        pvview = view.GetClientSideObject()
        renderer = pvview.GetRenderer(pvview.NON_COMPOSITED_RENDERER)
        legend = ScaleActor(self.meta.raw_config)
        renderer.AddActor(legend)

        def interaction_callback(*args, **kwargs):
            """Callback for interaction events."""
            self._copy_slice_camera(view)
            self._link_interaction()

        def fix_parallel_scale_callback(*args, **kwargs):
            """callback to fix the parallel scale on each render."""
            height = view.ViewSize[1] * self._active_subsampling_factor
            half_height = height / 2
            # ensures that the scale to a value to cause the
            # image to appear pixelated
            view.CameraParallelScale = max(half_height, view.GetActiveCamera().GetParallelScale())

            scale = self._active_subsampling_factor * view.CameraParallelScale / half_height
            legend.update_scale(scale)

        view.GetInteractor().AddObserver('InteractionEvent', interaction_callback)
        view.SMProxy.AddObserver('StartEvent', fix_parallel_scale_callback)
        return view

    def create_3d_view(self):
        view = self.create_render_view()
        view.CameraPosition = [1,1,1]

        def interaction_callback(*args, **kwargs):
            """Callback for interaction events."""
            self._link_interaction()

        view.GetInteractor().AddObserver('InteractionEvent', interaction_callback)
        return view

    def create_widget(self):
        self.update_client_state()

        with self.layout.viewport:
            with vuetify.VRow(v_if=f'{self._id}_no_maximized || {self.id}_max_row == 0', no_gutters=True, classes="grow"):
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 0'):
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[0] = self.create_slice_view(0)
                        self._html_views[0] = self.create_html_view(self._views[0])
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 1'):
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[1] = self.create_slice_view(1)
                        self._html_views[1] = self.create_html_view(self._views[1])

            with vuetify.VRow(v_if=f'{self._id}_no_maximized || {self.id}_max_row == 0', no_gutters=True, classes="shrink"):
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 0'):
                    self._ui_builder.slice_slider(self, 0)
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 0', cols="auto"):
                    self._ui_builder.maximize_button(self, 0, 0)
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 1'):
                    self._ui_builder.slice_slider(self, 1)
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 1', cols="auto"):
                    self._ui_builder.maximize_button(self, 0, 1)

            with vuetify.VRow(v_if=f'{self._id}_no_maximized || {self.id}_max_row == 1', no_gutters=True, classes="grow"):
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 0'):
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[2] = self.create_slice_view(2)
                        self._html_views[2] = self.create_html_view(self._views[2])
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 1'):
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[3] = self.create_3d_view()
                        self._html_views[3] = self.create_html_view(self._views[3])

            with vuetify.VRow(v_if=f'{self._id}_no_maximized || {self.id}_max_row == 1', no_gutters=True, classes="shrink"):
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 0'):
                    self._ui_builder.slice_slider(self, 2)
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 0', cols="auto"):
                    self._ui_builder.maximize_button(self, 1, 0)
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 1'):
                    vuetify.VContainer(classes="fill-height")
                with vuetify.VCol(v_if=f'{self._id}_no_maximized || {self.id}_max_col == 1', cols="auto"):
                    self._ui_builder.maximize_button(self, 1, 1)

        # add buttons to the button bar of the bottom of the view
        with self.layout.button_bar:
            if not self._force_outer_slices:
                with vuetify.VCol(cols='auto'):
                    self._ui_builder.toggle_button(self, var='show_inner_slices', off_icon='mdi-border-outside', on_icon='mdi-border-inside',
                      on_text='Show outer faces', off_text='Show inner slices')
            with vuetify.VCol(cols='auto'):
                self._ui_builder.toggle_button(self, var='full_res', on_icon='mdi-quality-high', off_icon='mdi-quality-low',
                    off_text='Show full resolution', on_text='Show low resolution',
                    click=self.toggle_full_res)
            with vuetify.VCol(cols='auto'):
                with vuetify.VTooltip(left=True):
                    with vuetify.Template(v_slot_activator="{on, attrs}"):
                        vuetify.VIcon("mdi-fit-to-screen",
                        click=self.reset_cameras,
                        classes="mr-4",
                        v_bind="attrs",
                        v_on="on",
                        __properties=[("v_bind", "v-bind"), ("v_on", "v-on")])
                    html.Pre("Reset zoom for all views")

    def update_pipeline(self):
        # update domains based on current dataset.
        log.info(f'{self.id}: updating pipeline')
        ext = self.producer.GetDataInformation().GetExtent()
        for axis in range(3):
            self._state[f'slice_min_{axis}'] = ext[axis*2]
            self._state[f'slice_max_{axis}'] = ext[axis*2+1]
            if self._state[f'slice_{axis}'] == -1:
                # only set if not initialized. this avoids overriding user chosen values
                self._state[f'slice_{axis}'] = ext[axis*2] + ext[axis*2+1] // 2

        # create the pipeline if it doesn't exist
        newly_created = self.create_pipeline()

        # update color map
        self.update_color_map()

        # update annotations
        self._annotations_source.Text = self.annotations_txt

        # update voi's for slices
        ext = self.producer.GetDataInformation().GetExtent()
        for axis in range(3):
            voi = list(ext)
            voi[axis*2] = voi[axis*2+1] = self._state[f'slice_{axis}']
            self._slices[axis].VOI = voi

        # update voi's for outer slices
        self.update_outer_slices(ext)

        # reset cameras
        if newly_created:
            self.reset_cameras()

        self.update_client_state()
        self.update_html_views()

    def update_outer_slices(self, ext):
        for axis in range(3):
            for side in range(2):
                voi = list(ext)
                voi[axis*2] = voi[axis*2+1] = ext[axis*2+side]
                self._outer_slices[axis*2+side].VOI = voi

    def create_pipeline(self):
        if hasattr(self, '_created_pipeline'):
            return False
        self._created_pipeline = True

        log.info(f'{self.id}: creating pipeline')

        # setup color map we'll use for this view
        self._lut = simple.GetColorTransferFunction(f'{self.id}_lut')
        self._lut.ApplyPreset('Grayscale', True)
        self._lut.RGBPoints = [0, 0.2, 0.2, 0.2, 1, 1, 1, 1]

        # create the pipeline
        self._color_mapyer = simple.ColorMappyer(Input=self.producer)
        self.create_outline_pipelines()
        for axis in range(3):
            self.create_slice_pipeline(axis)
        self.create_3d_pipeline()
        return True

    def reset_cameras(self):
        for view in self._views:
            if view is not None:
                self.reset_camera(view)
        self.update_html_views()

    def update_html_views(self):
        for html_view in self._html_views:
            if html_view is not None:
                html_view.update()

    def create_slice_pipeline(self, axis:int):
        """Creates the pipeline for the given axis."""
        ext = self.producer.GetDataInformation().GetExtent()
        log.info(f'{self.id}: creating slice pipeline for axis {axis} with extent {ext}')

        # create the slice
        slice = simple.ExtractVOI(Input=self._color_mapyer, VOI=ext)
        self._slices[axis] = slice

        # set the slice to the middle of the axis
        val = slice.VOI[axis*2] = slice.VOI[axis*2+1] = self._state[f'slice_{axis}']

        # create the slice representation
        sliceDisplay = simple.Show(slice, self._views[axis])
        simple.ColorBy(sliceDisplay, ('POINTS', self.get_scalar_name()))
        sliceDisplay.SetRepresentationType('Slice')
        sliceDisplay.LookupTable = self._lut
        sliceDisplay.MapScalars = self.get_map_scalars()

        # add annotation text
        text = simple.Text()
        text.Text = f'{CONSTANTS.AxisNames[axis]}: {self._active_subsampling_factor * val}'
        textDisplay = simple.Show(text, self._views[axis])
        textDisplay.Color = CONSTANTS.Colors[axis]
        textDisplay.FontSize = 16
        textDisplay.FontFamily = 'Arial'

        state = get_server().state
        @state.change(f'{self.id}_slice_{axis}')
        def slice_changed(**kwargs):
            val = kwargs[f'{self.id}_slice_{axis}']
            self._state[f'slice_{axis}'] = self._slices[axis].VOI[axis*2] = self._slices[axis].VOI[axis*2+1] = val
            setattr(self._outline, CONSTANTS.OutlinePropertyNames[axis], [val])
            text.Text = f'{CONSTANTS.AxisNames[axis]}: {self._active_subsampling_factor * val}'
            self.update_html_views()
            Base.propagate_changes_to_linked_views(self)

        return {
            'slice': slice,
            'text': text
        }

    def create_3d_pipeline(self):
        """Creates the 3D pipeline."""
        log.info(f'{self.id}: creating 3d pipeline')
        view = self._views[3]
        outline_display = simple.Show(self.producer, view)
        simple.ColorBy(outline_display, ('POINTS', self.get_scalar_name()))
        outline_display.SetRepresentationType('Outline')
        outline_display.LookupTable = self._lut

        # create 3 inner slice displays
        slice_displays = [None] * 3
        for axis in range(3):
            slice_display  = simple.Show(self._slices[axis], view)
            simple.ColorBy(slice_display, ('POINTS', self.get_scalar_name()))
            slice_display.SetRepresentationType('Slice')
            slice_display.LookupTable = self._lut
            slice_display.MapScalars = self.get_map_scalars()
            slice_displays[axis] = slice_display

        # create 6 outer slice displays
        ext = self.producer.GetDataInformation().GetExtent()
        for axis in range(3):
            for side in range(2):
                voi = simple.ExtractVOI(Input=self._color_mapyer, VOI=ext)
                voi.VOI[axis*2] = voi.VOI[axis*2+1] = ext[axis*2+side]
                slice_display = simple.Show(voi, view)
                simple.ColorBy(slice_display, ('POINTS', self.get_scalar_name()))
                slice_display.SetRepresentationType('Slice')
                slice_display.LookupTable = self._lut
                slice_display.MapScalars = self.get_map_scalars()
                simple.Hide(voi, view)
                self._outer_slices[axis*2+side] = voi
        self._update_3d_slice_visibility()

        # annotation
        text = simple.Text()
        text.Text = '[pending]'
        textDisplay = simple.Show(text, view)
        textDisplay.FontSize = 16
        textDisplay.FontFamily = 'Arial'
        textDisplay.Justification = 'Right'
        textDisplay.WindowLocation = 'Upper Right Corner'
        textDisplay.Opacity = 0.3
        self._annotations_source = text

        state = get_server().state
        @state.change(f'{self.id}_show_inner_slices')
        def show_inner_slices_changed(**kwargs):
            show_inner_slices = kwargs[f'{self.id}_show_inner_slices']
            self._state['show_inner_slices'] = show_inner_slices
            self._update_3d_slice_visibility()
            # update the html view for the 3d view
            self._html_views[3].update()
            Base.propagate_changes_to_linked_views(self)


    def _update_3d_slice_visibility(self):
        """Updates the visibility of the 3D slices."""
        show_inner_slices = self._state['show_inner_slices']
        view = self._views[3]
        for axis in range(3):
            simple.Show(self._slices[axis], view) if show_inner_slices else simple.Hide(self._slices[axis], view)
            for side in range(2):
                simple.Show(self._outer_slices[axis*2+side], view) if not show_inner_slices else simple.Hide(self._outer_slices[axis*2+side], view)


    def create_outline_pipelines(self):
        """Creates the outline pipelines."""
        self._outline = simple.ImageOutlineFilter(Input=self.producer)
        for view in self._views:
            outlineDisplay = simple.Show(self._outline, view)
            outlineDisplay.SetRepresentationType('Wireframe')
            outlineDisplay.MapScalars = self.get_map_scalars()
            outlineDisplay.ColorArrayName = ['POINTS', 'colors']
            outlineDisplay.LineWidth = 4

    def get_map_scalars(self):
        """Returns the map scalars value through LUT or not."""
        if self.meta.raw_config is not None and self.meta.raw_config.colormap is not None:
            return False
        if self._categories:
            return False
        return True

    def update_color_map(self):
        """Updates the color map."""
        log.info(f'{self.id}: updating color map')

        # get scalar bar in 3D view
        sb = simple.GetScalarBar(self._lut, self._views[3])
        sb.ComponentTitle = ''

        if self._categories:
            log.info(f'{self.id}: using categorical color map for categories (legacy)')
            self._lut.InterpretValuesAsCategories = True
            self._lut.AnnotationsInitialized = True
            annotations = []
            for seg, label in self._categories.items():
                annotations.append(str(seg))
                annotations.append(label)

            self._lut.Annotations = annotations
            count = min(11, max(3, len(self._categories)))
            self._lut.ApplyPreset(f'Brewer Diverging Spectral ({count})', True)

            colors = []
            scalars = []
            for seg in self._categories:
                color = [0.0, 0.0, 0.0]
                self._lut.GetClientSideObject().GetColor(seg, color)
                colors.append([color[0], color[1], color[2], 1.0])
                scalars.append(seg)

            # update scalar bar
            sb.Visibility = True
            sb.Title = 'segments'
            # update color mapyer
            # using this direct API call since the XML wrapping for this is broken
            self._color_mapyer.GetClientSideObject().SetColors(numpy.array(colors).flatten())
            self._color_mapyer.GetClientSideObject().SetScalars(scalars)
            assert self.get_map_scalars() == False
        elif self.meta.raw_config is not None and self.meta.raw_config.colormap is not None:
            log.info(f'{self.id}: using categorical color map (with color_mappyer)')
            self._lut.InterpretValuesAsCategories = False
            self._lut.AnnotationsInitialized = False
            sb.Visibility = False

            # update color mapyer
            # using this direct API call since the XML wrapping for this is broken
            self._color_mapyer.GetClientSideObject().SetColors(self.meta.raw_config.colormap['color'].flatten())
            self._color_mapyer.GetClientSideObject().SetScalars(self.meta.raw_config.colormap['scalar'].flatten())
            assert self.get_map_scalars() == False
        else:
            drange = self.producer.GetDataInformation().GetArrayInformation(self.get_scalar_name(), vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS).GetComponentRange(0)
            log.info(f'{self.id}: range: {drange}')
            self._lut.InterpretValuesAsCategories = False
            self._lut.ApplyPreset('Grayscale', True)
            self._lut.RGBPoints = [0, 0.2, 0.2, 0.2, 1, 1, 1, 1]
            self._lut.RescaleTransferFunction(drange[0], drange[1])
            sb.Visibility = True
            sb.Title = ''
            self._color_mapyer.Colors = []
            self._color_mapyer.Scalars = []

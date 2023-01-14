r"""quad view"""

from .base import Base
from vizer import utils

import os.path
import re

from paraview import simple, vtk
from trame.widgets import vuetify, paraview
from trame.app import get_server, asynchronous

log = utils.get_logger(__name__)

class CONSTANTS:
    Colors = [ [1., 0., 0.], [1., 1., 0.],  [0., 1., 0.] ]
    AxisNames = ['X', 'Y', 'Z']

class Quad(Base):
    """A quad view that renders the dataset in four views."""
    def __init__(self, filename, opts) -> None:
        super().__init__(filename, opts)
        self._views = [None, None, None, None]
        self._html_views = [None, None, None, None]
        self._outlines = [None, None, None]
        self._slices = [None, None, None]
        self._outer_slices = [None] * 6
        self._active_subsampling_factor = self.subsampling_factor

        # storing only data dependent state variables
        # that need to be updated on the client as well
        self._state = {}
        for axis in range(3):
            self._state[f'slice_min_{axis}'] = 0
            self._state[f'slice_max_{axis}'] = 0
            self._state[f'slice_{axis}'] = -1

        # next, state we want linked between views when requested.
        # self._state['full_res'] = self._full_res
        self._state['show_3d_slices'] = True
        self._state['full_res'] = False
        self._block_update = False

        # load the color categories, if present
        self.load_categories()

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
            log.info(f'propagating changes from {other.id} to {self.id} ...')
            self._state.update(other._state)
            self.update_client_state()

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
        meta_filename = f'{os.path.splitext(self.meta.filename)[0]}.txt'
        if not os.path.exists(meta_filename):
            return
        with open(meta_filename, 'r') as f:
            for line in f.readlines():
                line = line.strip().lower()
                if line.startswith('segorder:'):
                    line = line[len('segorder:'):]
                    regex = r"(?:\s*(?P<value>\d+)-(?P<text>[^,]+),?)"
                    matches = re.finditer(regex, line)
                    self._categories = dict([(int(m.group('value')), m.group('text')) for m in matches])
                    break
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

    def create_slice_view(self, axis:int):
        """Creates a slice view for the given axis."""
        view = self.create_render_view()
        # setup camera position
        pos = [ [10, 0, 0], [0, -10, 0], [0, 0, 10] ]
        up = [ [0, 0, 1], [0, 0, 1], [0, 1, 0] ]
        view.CameraPosition = pos[axis]
        view.CameraViewUp = up[axis]
        view.InteractionMode = '2D'

        def interaction_callback(*args, **kwargs):
            """Callback for interaction events."""
            self._copy_slice_camera(view)

        view.GetInteractor().AddObserver('InteractionEvent', interaction_callback)
        return view

    def create_3d_view(self):
        view = self.create_render_view()
        view.CameraPosition = [1,1,1]
        return view

    def create_widget(self):
        with self.layout.viewport:
            with vuetify.VRow(no_gutters=True, classes="grow"):
                with vuetify.VCol():
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[0] = self.create_slice_view(0)
                        self._html_views[0] = self.create_html_view(self._views[0])
                with vuetify.VCol():
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[1] = self.create_slice_view(1)
                        self._html_views[1] = self.create_html_view(self._views[1])

            with vuetify.VRow(no_gutters=True, classes="shrink"):
                with vuetify.VCol():
                    vuetify.VSlider(dense=True, hide_details=True,
                        min=(f'{self.id}_slice_min_0', 0), max=(f'{self.id}_slice_max_0', 0), v_model=(f'{self.id}_slice_0', 0))
                with vuetify.VCol():
                    vuetify.VSlider(dense=True, hide_details=True,
                        min=(f'{self.id}_slice_min_1', 0), max=(f'{self.id}_slice_max_1', 0), v_model=(f'{self.id}_slice_1', 0))

            with vuetify.VRow(no_gutters=True, classes="grow"):
                with vuetify.VCol():
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[2] = self.create_slice_view(2)
                        self._html_views[2] = self.create_html_view(self._views[2])
                with vuetify.VCol():
                    with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                        self._views[3] = self.create_3d_view()
                        self._html_views[3] = self.create_html_view(self._views[3])

            with vuetify.VRow(no_gutters=True, classes="shrink"):
                with vuetify.VCol():
                    vuetify.VSlider(dense=True, hide_details=True,
                        min=(f'{self.id}_slice_min_2', 0), max=(f'{self.id}_slice_max_2', 0), v_model=(f'{self.id}_slice_2', 0))
                with vuetify.VCol():
                    vuetify.VContainer(classes="fill-height")

        # add buttons to the button bar of the bottom of the view
        with self.layout.button_bar:
            with vuetify.VCol(cols='auto'):
                vuetify.VCheckbox(off_icon="mdi-border-outside", on_icon="mdi-border-inside",
                    v_model=(f'{self.id}_show_3d_slices', True), classes="mx-1 my-0", hide_details=True, dense=True)
            with vuetify.VCol(cols='auto'):
                vuetify.VCheckbox(on_icon="mdi-quality-high", off_icon="mdi-quality-low",
                    v_model=(f'{self.id}_full_res', self._state['full_res']),
                    classes="mx-1 my-0", hide_details=True, dense=True,
                    click=self.toggle_full_res)
            with vuetify.VCol(cols='auto'):
                # I use checkbox here to it has a consistent appearance with the other buttons
                vuetify.VCheckbox(on_icon="mdi-fit-to-screen", off_icon="mdi-fit-to-screen", hide_details=True, dense=True,
                    classes="mx-1 my-0",
                    click=self.reset_cameras)

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

        # update voi's for slices
        ext = self.producer.GetDataInformation().GetExtent()
        for axis in range(3):
            for side in range(2):
                voi = list(ext)
                voi[axis*2] = voi[axis*2+1] = ext[axis*2+side]
                self._outer_slices[axis*2+side].VOI = voi
            voi = list(ext)
            voi[axis*2] = voi[axis*2+1] = self._state[f'slice_{axis}']
            self._slices[axis].VOI = voi

        # reset cameras
        if newly_created:
            self.reset_cameras()

        self.update_client_state()

    def create_pipeline(self):
        if hasattr(self, '_created_pipeline'):
            return False
        self._created_pipeline = True

        log.info(f'{self.id}: creating pipeline')

        # setup color map we'll use for this view
        self._lut = simple.GetColorTransferFunction(f'{self.id}_lut')
        self._lut.ApplyPreset('Blue Orange (divergent)', True)

        # create the pipeline
        for axis in range(3):
            self.create_slice_pipeline(axis)
        self.create_3d_pipeline()
        self.create_outline_pipelines()
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
        slice = simple.ExtractSubset(Input=self.producer, VOI=ext)
        self._slices[axis] = slice

        # set the slice to the middle of the axis
        val = slice.VOI[axis*2] = slice.VOI[axis*2+1] = self._state[f'slice_{axis}']

        # create the slice representation
        sliceDisplay = simple.Show(slice, self._views[axis])
        simple.ColorBy(sliceDisplay, ('POINTS', 'ImageFile'))
        sliceDisplay.SetRepresentationType('Slice')
        sliceDisplay.LookupTable = self._lut

        # add annotation text
        text = simple.Text()
        text.Text = f'{CONSTANTS.AxisNames[axis]} Slice {self._active_subsampling_factor * val}'
        textDisplay = simple.Show(text, self._views[axis])
        textDisplay.Color = CONSTANTS.Colors[axis]

        state = get_server().state
        @state.change(f'{self.id}_slice_{axis}')
        def slice_changed(**kwargs):
            val = kwargs[f'{self.id}_slice_{axis}']
            # if self._state[f'slice_{axis}'] == val:
            #     return
            self._state[f'slice_{axis}'] = self._slices[axis].VOI[axis*2] = self._slices[axis].VOI[axis*2+1] = val
            text.Text = f'{CONSTANTS.AxisNames[axis]} Slice {self._active_subsampling_factor * val}'
            self.update_html_views()
            Base.propagate_changes_to_linked_views(self)


    def create_3d_pipeline(self):
        """Creates the 3D pipeline."""
        log.info(f'{self.id}: creating 3d pipeline')
        view = self._views[3]
        outline_display = simple.Show(self.producer, view)
        simple.ColorBy(outline_display, ('POINTS', 'ImageFile'))
        outline_display.SetRepresentationType('Outline')
        outline_display.LookupTable = self._lut

        # create 3 inner slice displays
        slice_displays = [None] * 3
        for axis in range(3):
            slice_display  = simple.Show(self._slices[axis], view)
            simple.ColorBy(slice_display, ('POINTS', 'ImageFile'))
            slice_display.SetRepresentationType('Slice')
            slice_display.LookupTable = self._lut
            slice_displays[axis] = slice_display

        # create 6 outer slice displays
        ext = self.producer.GetDataInformation().GetExtent()
        for axis in range(3):
            for side in range(2):
                voi = simple.ExtractSubset(Input=self.producer, VOI=ext)
                voi.VOI[axis*2] = voi.VOI[axis*2+1] = ext[axis*2+side]
                slice_display = simple.Show(voi, view)
                simple.ColorBy(slice_display, ('POINTS', 'ImageFile'))
                slice_display.SetRepresentationType('Slice')
                slice_display.LookupTable = self._lut
                simple.Hide(voi, view)
                self._outer_slices[axis*2+side] = voi

        state = get_server().state
        @state.change(f'{self.id}_show_3d_slices')
        def show_3d_slices_changed(**kwargs):
            show_slices = kwargs[f'{self.id}_show_3d_slices']
            # if self._state['show_3d_slices'] == show_slices:
            #     return
            self._state['show_3d_slices'] = show_slices
            for axis in range(3):
                simple.Show(self._slices[axis], view) if show_slices else simple.Hide(self._slices[axis], view)
                for side in range(2):
                    simple.Show(self._outer_slices[axis*2+side], view) if not show_slices else simple.Hide(self._outer_slices[axis*2+side], view)

            # update the html view for the 3d view
            self._html_views[3].update()
            Base.propagate_changes_to_linked_views(self)

    def create_outline_pipelines(self):
        for axis in range(3):
            self._outlines[axis] = simple.Outline(Input=self._slices[axis])

        for view in self._views:
            for axis in range(3):
                outlineDisplay = simple.Show(self._outlines[axis], view)
                outlineDisplay.SetRepresentationType('Outline')
                outlineDisplay.AmbientColor = outlineDisplay.DiffuseColor = CONSTANTS.Colors[axis]
                outlineDisplay.LineWidth = 2

    def update_color_map(self):
        """Updates the color map."""
        log.info(f'{self.id}: updating color map')
        if self._categories:
            self._lut.InterpretValuesAsCategories = True
            self._lut.AnnotationsInitialized = True
            annotations = []
            for seg, label in self._categories.items():
                annotations.append(str(seg))
                annotations.append(label)
            self._lut.Annotations = annotations
            count = min(11, max(3, len(self._categories)))
            self._lut.ApplyPreset(f'Brewer Diverging Spectral ({count})', True)
        else:
            drange = self.producer.GetDataInformation().GetArrayInformation('ImageFile', vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS).GetComponentRange(0)
            log.info(f'{self.id}: range: {drange}')
            self._lut.InterpretValuesAsCategories = False
            self._lut.RescaleTransferFunction(drange[0], drange[1])

        # show scalar bar in 3D view
        sb = simple.GetScalarBar(self._lut, self._views[3])
        sb.Visibility = True
        sb.Title = 'segments' if self._categories else ''
        sb.ComponentTitle = ''
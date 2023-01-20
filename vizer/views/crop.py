from .quad import Base, Quad, UIBuilder, CONSTANTS
from vizer import utils

from trame.widgets import vuetify, paraview
from trame.app import get_server, asynchronous

log = utils.get_logger(__name__)

class CropUIBuilder(UIBuilder):
    def slice_slider(self, view, axis):
        vuetify.VRangeSlider(
            v_model=(f'{view.id}_slice_range_{axis}', [0, 1]),
            min=(f'{view.id}_slice_min_{axis}', 0), max=(f'{view.id}_slice_max_{axis}', 0),
            hide_details=True, dense=True,
            class_='mx-1 my-0',
            # v_on='input',
            # v_on_input=f'update_slice("{view.id}", "{axis}")'
        )

class Crop(Quad):
    """extends Quad to add cropping functionality."""

    def __init__(self, meta, opts):
        super().__init__(meta, opts, force_outer_slices=True, use_vlayout=True,
            ui_builder=CropUIBuilder())
        for axis in range(3):
            self.state[f'slice_range_{axis}'] = [-1, -1]

    def create_widget(self):
        """creates the widget for this view."""
        return super().create_widget()

    def toggle_full_res(self):
        full_res = not self.state['full_res']
        for axis in range(3):
            val = list(self.state[f'slice_range_{axis}'])
            if full_res:
                val[0] = val[0] * self.subsampling_factor
                val[1] = val[1] * self.subsampling_factor
            else:
                val[0] = val[0] // self.subsampling_factor
                val[1] = val[1] // self.subsampling_factor
            self.state[f'slice_range_{axis}'] = tuple(val)
        return super().toggle_full_res()

    def update_pipeline(self):
        ext = self.producer.GetDataInformation().GetExtent()
        for axis in range(3):
            # only set if not initialized. this avoids overriding user chosen values
            if self.state[f'slice_range_{axis}'] == [-1, -1]:
                self.state[f'slice_range_{axis}'] = ext[axis*2:axis*2+2]
        return super().update_pipeline()

    def create_slice_pipeline(self, axis: int):
        """creates the slice pipeline for the given axis.""" 
        locals = super().create_slice_pipeline(axis)

        state = get_server().state
        @state.change(f'{self.id}_slice_range_{axis}')
        def update_slice_range(**kwargs):
            val = kwargs.get(f'{self.id}_slice_range_{axis}', None)
            old_val = self.state[f'slice_range_{axis}']
            slice_val = val[0] if val[1] == old_val[1] else val[1]

            locals['slice'].VOI[axis*2] = locals['slice'].VOI[axis*2+1] = slice_val
            setattr(self._outline, CONSTANTS.OutlinePropertyNames[axis], val)

            self.state[f'slice_range_{axis}'] = val
            scaled_val = [self._active_subsampling_factor * val[0], self._active_subsampling_factor * val[1]]
            locals['text'].Text = f'{CONSTANTS.AxisNames[axis]}: {scaled_val[0]} - {scaled_val[1]}'

            ext = [*self.state['slice_range_0'], *self.state['slice_range_1'], *self.state['slice_range_2']]
            self.update_outer_slices(ext)
            self.update_html_views()
            Base.propagate_changes_to_linked_views(self)

from .quad import Quad
from vizer import utils

from trame.widgets import vuetify, paraview

log = utils.get_logger(__name__)

class Crop(Quad):
    """extends Quad to add cropping functionality."""

    def __init__(self, meta, opts):
        super().__init__(meta, opts, force_outer_slices=True, use_vlayout=True)

    def create_widget(self):
        """creates the widget for this view."""
        with self.layout.button_bar:
            with vuetify.VCol(cols='auto'):
                vuetify.VCheckbox(off_icon="mdi-crop", on_icon="mdi-selection-drag",
                    v_model=(f'{self.id}_crop', False), classes="mx-1 my-0", hide_details=True, dense=True)

        return super().create_widget()
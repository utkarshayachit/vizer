from abc import ABC, abstractmethod

from paraview import simple, vtk
from trame.widgets import vuetify, paraview, html
from trame.app import get_server

from vizer import utils
import os.path
import gc
import weakref

log = utils.get_logger(__name__)

class Layout(vuetify.VCard):
    def __init__(self, title=None, close=None):
        super().__init__(dark=True,
            rounded='lg',
            outlined=True,
            classes="grow d-flex flex-column flex-nowrap pa-0 ma-0",
            style="overflow: hidden")

        with self:
            with vuetify.VRow(no_gutters=True, classes="shrink grey"):
                with vuetify.VCol():
                    with vuetify.VSystemBar(title, dense=True, classes="grey ma-0 px-2 py-0", style="text-overflow: ellipsis; overflow: hidden; white-space: nowrap;"):
                        if close is not None:
                            vuetify.VSpacer()
                            with vuetify.VTooltip(left=True):
                                with vuetify.Template(v_slot_activator="{on, attrs}"):
                                    vuetify.VIcon("mdi-close", click=close,
                                                  v_bind="attrs",
                                                  v_on="on",
                                                  __properties=[('v_bind', 'v-bind'), ('v_on', 'on')])
                                html.Pre("Close")

            with vuetify.VRow(no_gutters=True, classes="grow"):
                self.viewport = vuetify.VCard(dark=True,
                    # remove the elevation
                    flat=True,
                    # remove the border radius
                    tile=True,
                    classes="grow d-flex flex-column flex-nowrap pa-0 ma-0 fill-height", style="overflow: hidden")
            self.button_bar = vuetify.VRow(no_gutters=True, classes="shrink grey")
            with self.button_bar:
                with vuetify.VCol():
                    vuetify.VTextField(v_model=f'status_{id(self)}', hide_details=True, flat=True, readonly=True,
                        dense=True, dark=True, disabled=True,
                        classes="ma-0 px-2 py-0")
                vuetify.VSpacer()

            # this is used to show a busy indicator; a modal text dialog is popped up to avoid
            # any further interaction with the UI until the task is completed.
            with vuetify.VDialog(v_model=(f'busy_{id(self)}', False), max_width="150", persistent=True):
                with vuetify.VCard(dark=True, style="overflow: hidden"):
                    with vuetify.VCardText(classes="pa-5"):
                        with vuetify.VRow():
                            vuetify.VCol("updating ...")

            # this is used to show a modal dialog with arbitrary content
            self.dialog = vuetify.VDialog(v_model=(f'modal_{id(self)}', False), fullscreen=True)

        self.set_status('initializing ...')

    def set_status(self, status):
        state = get_server().state
        state[f'status_{id(self)}'] = status
        state.flush()

    def start_busy(self):
        state = get_server().state
        state[f'busy_{id(self)}'] = True
        state.flush()

    def stop_busy(self):
        state = get_server().state
        state[f'busy_{id(self)}'] = False
        state.flush()

    def show_dialog(self):
        state = get_server().state
        state[f'modal_{id(self)}'] = True
        state.flush()

    def hide_dialog(self):
        state = get_server().state
        state[f'modal_{id(self)}'] = False
        state.flush()

class Base(ABC):
    _propagating = False
    _all_views_refs = set()
    _counter = 0

    @classmethod
    def _next_id(cls):
        cls._counter += 1
        return cls._counter

    def __init__(self, meta, opts) -> None:
        super().__init__()
        self._id = f'{self.__class__.__name__.lower()}_{Base._next_id()}'
        self._filename = meta.filename
        self._meta = meta
        self._opts = opts
        self._producer = simple.PVTrivialProducer()
        self._all_views_refs.add(weakref.ref(self))

    def __del__(self):
        simple.Delete(self._producer)
        self._producer = None

    @classmethod
    def get_linked_views(cls, view):
        """returns the list of linked views."""
        if view is None or view.opts.link_views is False:
            return []
        return [ref() for ref in cls._all_views_refs if ref() is not None and ref() != view and isinstance(ref(), type(view))]

    @classmethod
    def propagate_changes_to_linked_views(cls, view):
        """propagate changes to linked views."""
        if view is None or view.opts.link_views is False or cls._propagating or view is None:
            return
        cls._propagating = True
        for tview in cls.get_linked_views(view):
            tview.copy_state_from(view)
        cls._propagating = False

    @property
    def id(self):
        return self._id

    @property
    def filename(self):
        """returns the filename associated with this view."""
        return self._filename

    @property
    def opts(self):
        """returns the options associated with this view."""
        return self._opts

    @property
    def meta(self):
        """returns the metadata associated with this view's dataset."""
        return self._meta

    @property
    def layout(self):
        if not hasattr(self, '_layout'):
            self._layout = Layout(os.path.basename(self.filename))
            self.set_status('initializing ...')
        return self._layout

    @property
    def widget(self):
        """returns the widget to be displayed in the UI."""
        if not hasattr(self, '_widget_created'):
            self._widget_created = True
            self.create_widget()
        return self.layout

    @abstractmethod
    def create_widget(self):
        """create a new widget to be displayed in the UI."""
        raise RuntimeError('not implemented')

    @property
    def producer(self):
        """returns the data producer that will be used to feed the view."""
        return self._producer

    @property
    def dataset(self):
        """returns the dataset associated with this view."""
        return self.producer.GetClientSideObject().GetOutputDataObject(0)

    @property
    def subsampling_factor(self):
        return max(self._opts.subsampling_factor, 1)

    async def load_dataset(self, async_only=False):
        """load/reload the dataset associated with this view."""
        log.info(f'{self.id}: loading dataset for {self.filename}')
        self.set_status('loading dataset ...')
        awaitable = async_only
        if not async_only or not self.meta.is_raw():
            dataset, awaitable = self.meta.sync_read_dataset(self.opts)
            self.set_dataset(dataset)
            del dataset # release reference
        if awaitable:
            self.layout.start_busy()
            self.set_status('loading dataset asynchronously ...')
            self.set_dataset(await self.meta.async_read_dataset(self.opts))
            self.layout.stop_busy()
        self.set_status('ready')

    def set_dataset(self, dataset):
        """set the dataset that will be used to feed the view. the pipeline
        will be updated to use the new producer."""
        log.info(f'{self.id}: setting dataset for {self.filename}')
        dataset = self.subsample(dataset)
        gc.collect()

        self._producer.GetClientSideObject().SetOutput(dataset)
        if dataset and dataset.GetExtentType() == vtk.VTK_3D_EXTENT:
            self._producer.WholeExtent = dataset.GetExtent()
            log.info(f'{self.id}: whole extent: {self._producer.WholeExtent}')
        
        self._producer.MarkModified(self._producer)
        self._producer.UpdatePipeline()
        self.update_pipeline()
        gc.collect()

    @abstractmethod
    def update_pipeline(self):
        """update the pipeline to use the current producer."""
        pass

    def create_render_view(self):
        """create a render view to be used by the view."""
        view = simple.CreateRenderView()
        # change the background color to match the theme
        view.Background = [0.12, 0.12, 0.12]
        view.UseColorPaletteForBackground = False
        return view

    def create_html_view(self, render_view):
        """create a html view to be used by the view."""
        html_view = paraview.VtkRemoteView(render_view,
            interactive_ratio=1.0,
            ref=f'view{render_view.GetGlobalID()}')
        return html_view

    def reset_camera(self, view):
        """reset the camera of the view."""
        view.ResetCamera(True)
        view.StillRender()
        view.CenterOfRotation = view.CameraFocalPoint.GetData()

    def set_status(self, status):
        """set the status of the view."""
        log.info(f'set_status: {status}')
        self.layout.set_status(status)

    def subsample(self, dataset):
        if self.subsampling_factor > 1 and dataset and dataset.GetExtentType() == vtk.VTK_3D_EXTENT:
            log.info(f'{self.id}: subsampling dataset by a factor of {self.opts.subsampling_factor}')
            from vtkmodules.vtkAcceleratorsVTKmFilters import vtkmExtractVOI as vtkExtractVOI
            subsampler = vtkExtractVOI()
            subsampler.SetInputData(dataset)
            subsampler.SetVOI(dataset.GetExtent())
            subsampler.SetSampleRate(self.opts.subsampling_factor, self.opts.subsampling_factor, self.opts.subsampling_factor)
            subsampler.IncludeBoundaryOff()
            subsampler.Update()
            dataset = subsampler.GetOutputDataObject(0)
        return dataset

    def get_scalar_name(self):
        """returns the scalar array name"""
        scalars = self.producer.GetDataInformation().GetPointDataInformation().GetAttributeInformation(
            vtk.vtkDataSetAttributes.SCALARS)
        return scalars.GetName() if scalars is not None else None

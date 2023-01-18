r"""A simple view that renders a single dataset."""

from .base import Base

from paraview import simple

class Simple(Base):
    """A simple view that renders a single dataset."""
    def __init__(self, meta, opts):
        super().__init__(meta, opts)

    @staticmethod
    def can_show(meta):
        """returns true if this view can show the dataset from the file"""
        return False if meta.is_empty() else True

    def create_widget(self):
        with self.layout.viewport:
            self._render_view = self.create_render_view()
            self._html_view = self.create_html_view(self._render_view)
        

    def update_pipeline(self):
        self.create_pipeline()
        # since the producer may have changed, we need to update the input
        # of the pass through filter -- our entry point to the pipeline
        self.producer.UpdatePipeline()
        self.reset_camera(self._render_view)

    def create_pipeline(self):
        if not hasattr(self, '_display'):
            self._display = simple.Show(self.producer, self._render_view)
        

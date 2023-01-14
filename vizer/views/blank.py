r"""view to show when data cannot be loaded or view cannot be created."""

from .base import Base

from paraview import simple

class Blank(Base):
    """view to show when data cannot be loaded or view cannot be created."""
    def __init__(self, meta, opts):
        super().__init__(meta, opts)

    @staticmethod
    def can_show(meta):
        """returns true if this view can show the dataset from the file"""
        return True

    def create_widget(self):
        with self.layout.viewport:
            super().set_status('cannot load data')

    def update_pipeline(self):
        pass

    async def load_dataset(self, async_only=False):
        pass
        
    def set_status(self, status):
        pass
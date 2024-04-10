from typing import Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, Center


@functional_transform('normalize_scale')
class NormalizeScale(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self):
        self.center = Center()

    def forward(self, 
                data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        data = self.center(data)
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                scale = (1 / store.pos.abs().max()) * 0.999999
                store.pos = store.pos * scale

        return data
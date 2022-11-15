from util.datasets import SatelliteDataset


class QFabricDataset(SatelliteDataset):
    CHANGE_TYPES = ['No Change', 'Residential', 'Commercial', 'Industrial',
                    'Road', 'Demolition', 'Mega Projects']
    CHANGE_STATUS = ['No Change', 'Prior Construction', 'Greenland',
                     'Land Cleared', 'Excavation', 'Materials Dumped',
                     'Construction Started', 'Construction Midway',
                     'Construction Done', 'Operational']

    def __init__(self):
        super(QFabricDataset, self).__init__(in_c=3)


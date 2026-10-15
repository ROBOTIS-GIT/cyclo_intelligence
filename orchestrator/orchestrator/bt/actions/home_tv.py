"""TV power operation at the simulation TV control station."""
from orchestrator.bt.actions.home_meal import HomeMeal


class HomeTV(HomeMeal):
    SERVICE_PREFIX = '/simulation/home_tv/'
    OPERATIONS = ('turn_on', 'turn_off')

    def __init__(self, node, operation: str = 'turn_off', timeout: float = 15.0,
                 check_only: bool = False):
        super().__init__(node, operation, timeout, check_only)

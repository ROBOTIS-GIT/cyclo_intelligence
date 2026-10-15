"""Simulation laundry prop transitions after a JointControl reach."""
from orchestrator.bt.actions.home_meal import HomeMeal


class HomeLaundry(HomeMeal):
    SERVICE_PREFIX = '/simulation/home_laundry/'
    OPERATIONS = ('pickup_basket', 'load_washer', 'start_wash')

    def __init__(self, node, operation: str = 'pickup_basket', timeout: float = 15.0,
                 check_only: bool = False):
        super().__init__(node, operation, timeout, check_only)

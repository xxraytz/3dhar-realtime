import logging
from collections import deque
from enum import Enum

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MState(Enum):
    START_MONITORING = "start"
    APPEND_MONITORING = "start"
    END_MONITORING = "end"
    GET_PERSON_ID = "get_id"
    CHECK_STATUS = "check_status"
    INIT_MONITORING = "initialization"


def identify(qm, config):
    skeleton_sequence = deque(maxlen=300)

    while True:

        data = qm.get()
        if isinstance(data, tuple):
            state = data[0]
        else:
            state = data

        if state.value is MState.CHECK_STATUS.value:
            logger.info(state.value)
            print(state.value)
        elif state.value is MState.INIT_MONITORING:
            # Здесь необходимо создать массив потоков скелетных данных в зависимости от N, для Kinect N = 6,
            # следовательно data[1] должно быть равно 6
            pass
        elif state.value is MState.END_MONITORING.value:
            break
        elif state.value is MState.APPEND_MONITORING.value:
            skeletons_frame = data[1]
            for skeleton in skeletons_frame:

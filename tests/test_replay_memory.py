from core.replay_memory import ReplayMemory, Transition
import numpy as np


def test_get_latest():
    mem = ReplayMemory(4250, "cpu")
    for i in range(5000):
        transition = Transition(i, 'Old', None, None, None)
        mem.add(*transition)

    assert (len(mem) == 4250)

    # other.position >= latest
    latest = []
    for i in range(2000):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.add(*transition)

    assert (len(mem) == 4250)

    returned_latest = mem.get_latest(2000)
    assert (len(returned_latest) == 2000)
    for i, trans in enumerate(latest):
        assert (trans.state == returned_latest[i].state)
        assert (trans.action == returned_latest[i].action)

    # other.position < latest and buffer is full
    latest = []
    for i in range(2000):
        transition = Transition(i, 'New2', None, None, None)
        latest.append(transition)
        mem.add(*transition)

    assert (len(mem) == 4250)
    assert (mem.position < 2000)

    returned_latest = mem.get_latest(2000)
    assert (len(returned_latest) == 2000)
    for i, trans in enumerate(latest):
        assert (trans.state == returned_latest[i].state)
        assert (trans.action == returned_latest[i].action)


def test_get_latest_from_small_capacity():
    mem = ReplayMemory(1150, "cpu")
    for i in range(3000):
        transition = Transition(i, 'Old', None, None, None)
        mem.add(*transition)

    assert (len(mem) == 1150)

    latest = []
    for i in range(2000):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.add(*transition)

    returned_latest = mem.get_latest(2000)
    assert (len(returned_latest) == 1150)
    for i, trans in enumerate(latest[-1150:]):
        assert (trans.state == returned_latest[i].state)
        assert (trans.action == returned_latest[i].action)


def test_get_latest_from_buffer_not_full():
    mem = ReplayMemory(2150, "cpu")

    latest = []
    for i in range(1300):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.add(*transition)

    returned_latest = mem.get_latest(1300)
    assert (len(returned_latest) == 1300)
    for i, trans in enumerate(latest):
        assert (trans.state == returned_latest[i].state)
        assert (trans.action == returned_latest[i].action)


def test_add_latest_from():
    mem = ReplayMemory(2150, "cpu")
    mem2 = ReplayMemory(3000, "cpu")

    latest = []
    for i in range(1300):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.add(*transition)

    mem2.add_latest_from(mem, 1000)
    assert (len(mem2) == 1000)
    for i, trans in enumerate(latest[-1000:]):
        assert (trans.state == mem2.memory[i].state)
        assert (trans.action == mem2.memory[i].action)


def test_shuffle():
    mem = ReplayMemory(2150, "cpu")

    for i in range(1300):
        transition = Transition(i, 'New', None, None, None)
        mem.add(*transition)

    mem.shuffle()
    returned_latest = mem.get_latest(1300)

    shuffled_states = []
    for i, trans in enumerate(returned_latest):
        shuffled_states.append(returned_latest[i].state[0][0])

    ordered_states = np.sort(shuffled_states)

    # Check the shuffled list is different
    assert (list(np.arange(1300)) != shuffled_states)

    # Check the ordered shuffled list is the same
    assert (list(np.arange(1300)) == list(ordered_states))


def test_add_content_of():
    mem = ReplayMemory(2150, "cpu")
    other = ReplayMemory(1560, "cpu")

    latest = []
    for i in range(2000):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        other.add(*transition)

    mem.add_content_of(other)

    for i, trans in enumerate(latest[-1560:]):
        assert (trans.state == mem.memory[i].state)

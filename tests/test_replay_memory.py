from core.replay_memory import ReplayMemory, Transition


def test_get_latest():
    mem = ReplayMemory(4250, "cpu")
    for i in range(5000):
        transition = Transition(i, 'Old', None, None, None)
        mem.push(transition)

    assert (len(mem) == 4250)

    # other.position >= latest
    latest = []
    for i in range(2000):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.push(transition)

    assert (len(mem) == 4250)

    returned_latest = mem.get_latest(2000)
    assert (len(returned_latest) == 2000)
    for i, trans in enumerate(latest):
        assert (trans is returned_latest[i])

    # other.position < latest and buffer is full
    latest = []
    for i in range(2000):
        transition = Transition(i, 'New2', None, None, None)
        latest.append(transition)
        mem.push(transition)

    assert (len(mem) == 4250)
    assert (mem.position < 2000)

    returned_latest = mem.get_latest(2000)
    assert (len(returned_latest) == 2000)
    for i, trans in enumerate(latest):
        assert (trans is returned_latest[i])


def test_get_latest_from_small_capacity():
    mem = ReplayMemory(1150, "cpu")
    for i in range(3000):
        transition = Transition(i, 'Old', None, None, None)
        mem.push(transition)

    assert (len(mem) == 1150)

    latest = []
    for i in range(2000):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.push(transition)

    returned_latest = mem.get_latest(2000)
    assert (len(returned_latest) == 1150)
    for i, trans in enumerate(latest[-1150:]):
        assert (trans is returned_latest[i])


def test_get_latest_from_buffer_not_full():
    mem = ReplayMemory(2150, "cpu")

    latest = []
    for i in range(1300):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.push(transition)

    returned_latest = mem.get_latest(1300)
    assert (len(returned_latest) == 1300)
    for i, trans in enumerate(latest):
        assert (trans is returned_latest[i])


def test_add_latest_from():
    mem = ReplayMemory(2150, "cpu")
    mem2 = ReplayMemory(3000, "cpu")

    latest = []
    for i in range(1300):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.push(transition)

    mem2.add_latest_from(mem, 1000)
    assert (len(mem2) == 1000)
    for i, trans in enumerate(latest[-1000:]):
        assert (trans is mem2.memory[i])


def test_shuffle():
    mem = ReplayMemory(2150, "cpu")

    latest = []
    for i in range(1300):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        mem.push(transition)

    returned_latest = mem.get_latest(1300)
    assert (len(returned_latest) == 1300)
    assert (returned_latest == latest)

    mem.shuffle()
    returned_latest = mem.get_latest(1300)
    assert (returned_latest != latest)


def test_add_content_of():
    mem = ReplayMemory(2150, "cpu")
    other = ReplayMemory(1560, "cpu")

    latest = []
    for i in range(2000):
        transition = Transition(i, 'New', None, None, None)
        latest.append(transition)
        other.push(transition)

    mem.add_content_of(other)
    latest_now = mem.get_latest(1560)
    assert (latest_now == latest[-1560:])

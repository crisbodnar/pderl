from core.agent import Archive


class Args:
    def __init__(self):
        self.archive_size = 100
        self.ns_k = 2


args = Args()


def test_add_bc():
    archive = Archive(args)
    assert archive.size() == 0
    assert archive.get_novelty([3, 3]) == 18
    archive.add_bc([1, 1])
    assert archive.size() == 1
    assert archive.get_novelty([3, 3]) == 8


def test_add_above_limit():
    archive = Archive(args)
    for i in range(101):
        archive.add_bc([0, 0])
    assert archive.size() == 100
    archive.add_bc([1, 1])
    assert archive.bcs[-1] == [1, 1]


def test_get_novelty():
    archive = Archive(args)
    archive.add_bc([4, 4])
    archive.add_bc([-5, 10])
    archive.add_bc([100, 40])
    archive.add_bc([-10, -10])
    assert archive.get_novelty([90, 60]) == 5516



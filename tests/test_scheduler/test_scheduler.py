"""算子调度器测试：依赖正确性、资源约束、优先级、异常与规模健壮性。"""
import pytest

from sut.scheduler import (
    CycleError,
    MissingDependencyError,
    Schedule,
    SchedulerError,
    Task,
    UnschedulableError,
    schedule,
)

pytestmark = pytest.mark.scheduler


class TestDependency:
    def test_respects_dependency_order(self):
        result = schedule([Task("b", deps=("a",)), Task("a")], resource_budget=10)
        assert result.order() == ["a", "b"]

    def test_diamond_dependency(self):
        tasks = [
            Task("root"), Task("left", deps=("root",)),
            Task("right", deps=("root",)), Task("sink", deps=("left", "right")),
        ]
        result = schedule(tasks, resource_budget=10)
        order = result.order()
        assert order.index("root") < order.index("left")
        assert order.index("root") < order.index("right")
        assert order.index("sink") > max(order.index("left"), order.index("right"))

    def test_chain_makespan_equals_depth(self):
        tasks = [Task("t1"), Task("t2", deps=("t1",)), Task("t3", deps=("t2",))]
        result = schedule(tasks, resource_budget=10)
        assert result.makespan() == 3


class TestResource:
    def test_budget_not_exceeded(self):
        tasks = [Task(f"t{i}", cost=3) for i in range(6)]
        result = schedule(tasks, resource_budget=7)
        for slot in result.slots:
            assert sum(next(t.cost for t in tasks if t.id == tid) for tid in slot) <= 7

    def test_independent_tasks_packed(self):
        """预算 10、任务 cost=5：两个任务应装入同槽。"""
        tasks = [Task("a", cost=5), Task("b", cost=5)]
        result = schedule(tasks, resource_budget=10)
        assert result.makespan() == 1 and result.slots[0] == ["a", "b"]

    def test_oversize_task_rejected(self):
        with pytest.raises(UnschedulableError, match="超过预算"):
            schedule([Task("huge", cost=11)], resource_budget=10)

    def test_invalid_budget(self):
        with pytest.raises(SchedulerError, match="预算"):
            schedule([Task("a")], resource_budget=0)


class TestPriority:
    def test_higher_priority_first_among_ready(self):
        tasks = [Task("low", priority=1), Task("high", priority=9), Task("mid", priority=5)]
        result = schedule(tasks, resource_budget=1)  # 每槽只能装一个
        assert result.order() == ["high", "mid", "low"]

    def test_tie_break_by_id(self):
        tasks = [Task("z", priority=5), Task("a", priority=5)]
        result = schedule(tasks, resource_budget=1)
        assert result.order() == ["a", "z"], "同优先级按 id 字典序稳定排序"

    def test_priority_does_not_break_deps(self):
        """高优先级任务依赖未调度的低优先级任务时，依赖仍优先。"""
        tasks = [Task("dep", priority=0), Task("star", priority=100, deps=("dep",))]
        result = schedule(tasks, resource_budget=1)
        assert result.order()[0] == "dep"


class TestErrors:
    def test_cycle_detected(self):
        tasks = [Task("a", deps=("b",)), Task("b", deps=("a",))]
        with pytest.raises(CycleError):
            schedule(tasks, resource_budget=10)

    def test_self_cycle(self):
        with pytest.raises(CycleError, match="自身"):
            schedule([Task("a", deps=("a",))], resource_budget=10)

    def test_missing_dependency(self):
        with pytest.raises(MissingDependencyError):
            schedule([Task("a", deps=("ghost",))], resource_budget=10)

    def test_duplicate_id(self):
        with pytest.raises(SchedulerError, match="重复"):
            schedule([Task("a"), Task("a")], resource_budget=10)

    def test_nonpositive_cost(self):
        with pytest.raises(SchedulerError, match="cost"):
            schedule([Task("a", cost=0)], resource_budget=10)


class TestScale:
    @pytest.mark.parametrize("n", [50, 200], ids=["n50", "n200"])
    def test_random_dag_schedules_validly(self, n):
        """随机生成大规模 DAG，断言调度结果的依赖与资源约束全部满足。"""
        import random

        rng = random.Random(n)
        tasks = []
        for i in range(n):
            deps = tuple(rng.sample(range(i), k=min(i, rng.randint(0, 3)))) \
                if i else ()
            tasks.append(Task(f"task{i:04d}", cost=rng.randint(1, 3),
                              priority=rng.randint(0, 9), deps=tuple(
                                  f"task{j:04d}" for j in deps)))
        result = schedule(tasks, resource_budget=6)
        pos = {tid: idx for idx, tid in enumerate(result.order())}
        for t in tasks:
            for dep in t.deps:
                assert pos[dep] < pos[t.id], f"{dep} 应先于 {t.id}"
            assert t.cost <= 6
        for slot in result.slots:
            cost = sum(next(t.cost for t in tasks if t.id == tid) for tid in slot)
            assert cost <= 6

    def test_schedule_is_deterministic(self):
        tasks = [Task(f"t{i}", cost=1, priority=(i * 7) % 5) for i in range(20)]
        r1 = schedule(tasks, resource_budget=3)
        r2 = schedule(tasks, resource_budget=3)
        assert r1.order() == r2.order(), "同输入调度结果必须确定性一致"